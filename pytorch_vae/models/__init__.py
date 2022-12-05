from pytorch_vae.models.base import *
from pytorch_vae.models.vanilla_vae import *
from pytorch_vae.models.gamma_vae import *
from pytorch_vae.models.beta_vae import *
from pytorch_vae.models.wae_mmd import *
from pytorch_vae.models.cvae import *
from pytorch_vae.models.hvae import *
from pytorch_vae.models.vampvae import *
from pytorch_vae.models.iwae import *
from pytorch_vae.models.dfcvae import *
from pytorch_vae.models.mssim_vae import MSSIMVAE
from pytorch_vae.models.fvae import *
from pytorch_vae.models.cat_vae import *
from pytorch_vae.models.joint_vae import *
from pytorch_vae.models.info_vae import *
# from pytorch_vae.models.twostage_vae import *
from pytorch_vae.models.lvae import LVAE
from pytorch_vae.models.logcosh_vae import *
from pytorch_vae.models.swae import *
from pytorch_vae.models.miwae import *
from pytorch_vae.models.vq_vae import *
from pytorch_vae.models.betatc_vae import *
from pytorch_vae.models.dip_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {'HVAE':HVAE,
              'LVAE':LVAE,
              'IWAE':IWAE,
              'SWAE':SWAE,
              'MIWAE':MIWAE,
              'VQVAE':VQVAE,
              'DFCVAE':DFCVAE,
              'DIPVAE':DIPVAE,
              'BetaVAE':BetaVAE,
              'InfoVAE':InfoVAE,
              'WAE_MMD':WAE_MMD,
              'VampVAE': VampVAE,
              'GammaVAE':GammaVAE,
              'MSSIMVAE':MSSIMVAE,
              'JointVAE':JointVAE,
              'BetaTCVAE':BetaTCVAE,
              'FactorVAE':FactorVAE,
              'LogCoshVAE':LogCoshVAE,
              'VanillaVAE':VanillaVAE,
              'ConditionalVAE':ConditionalVAE,
              'CategoricalVAE':CategoricalVAE}
