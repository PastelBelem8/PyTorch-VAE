import os
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from pytorch_vae.models import *
from experiment import VAEXperiment
from lightning_lite.utilities.seed import seed_everything
from pytorch_vae.datasets import VAEDataset


def read_config(yaml_path: str):
    with open(yaml_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

def load_vae(filepath, model, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c', dest="config", metavar='FILE', help = 'path to the config file', default='../configs/vae.yaml')
    parser.add_argument("--data-config", '-d', dest="data_configs", metavar='FILE', help = 'path to the data config file', default='../configs/dataset/tiny-imagenet-200.yaml')
    parser.add_argument('--checkpoint',  '-ckpt', dest="checkpoint", metavar='FILE', help = 'path to the checkpoint file', default='../logs/VanillaVAE/version_0/checkpoints/last.ckpt')

    args = parser.parse_args()
    config = read_config(args.config)

    # It will override data definitions in model config files
    data_config = read_config(args.data_configs)
    data_config["data_params"]["output_filepath"] = True
    config.update(**data_config)
    print(config)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    # Load model
    load_vae(args.checkpoint, model, "cuda" if torch.cuda.is_available() else "cpu")
    experiment = VAEXperiment(model, config['exp_params'])

    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()

    # Task 1. For each training and test set example
    # - Get corresponding latent encoding z
    train_dataloader = data.train_dataloader()
    #val_dataloader = data.val_dataloader()

    generated_z = {}
    for images, _, paths in train_dataloader:

        mu, log_var = model.encode(images)
        zs = model.reparameterize(mu, log_var)

        for i, p in enumerate(paths):
            generated_z[p] = zs[i].detach().cpu()

    torch.save(generated_z, "../data/tiny-imagenet-200-zs/train.pkl")