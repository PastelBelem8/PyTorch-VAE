import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from pytorch_vae.models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_vae.datasets import VAEDataset
from pytorch_lightning.strategies import DDPStrategy

def read_config(yaml_path: str):
    with open(yaml_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument(
        '--config',  '-c', dest="config", metavar='FILE', help = 'path to the config file', default='../configs/vae.yaml')

    parser.add_argument(
        "--data-config", '-d', dest="data_configs", metavar='FILE', help = 'path to the data config file', default='../configs/dataset/tiny-imagenet-200.yaml'
    )

    args = parser.parse_args()
    config = read_config(args.config)

    # It will override data definitions in model config files
    data_config = read_config(args.data_configs)
    config.update(**data_config)
    print(config)

    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                name=config['model_params']['name'],)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                            config['exp_params'])

    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

    data.setup()
    runner = Trainer(
        accelerator="gpu",
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
                monitor= "val_loss",
                save_last= True),
            EarlyStopping(monitor="val_loss", mode="min")
        ],
        strategy=DDPStrategy(find_unused_parameters=False),
        **config['trainer_params'],
    )


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)