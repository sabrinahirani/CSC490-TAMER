from pytorch_lightning.utilities.cli import LightningCLI
# import torch.backends

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER


cli = LightningCLI(
          LitTAMER,
          HMEDatamodule,
          save_config_overwrite=True,
          trainer_defaults={
              'precision': 32,  # enable mixed precision
              'accumulate_grad_batches': 2,  # reduce memory spikes
              'auto_scale_batch_size': 'binsearch',  # dynamically reduce batch size
              'gpus': 1,
          }
)