import torch
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

torch.cuda.empty_cache()

from pytorch_lightning import Trainer

def main():
    cli = LightningCLI(
        LitTAMER,
        HMEDatamodule,
        save_config_overwrite=True,
        trainer_defaults={
            'precision': 16,  # enable mixed precision
            'accumulate_grad_batches': 2,  # reduce memory spikes
            'callbacks': [
                ModelCheckpoint(monitor='val_loss', save_top_k=3)
            ]
        }
    )

if __name__ == "__main__":
    main()