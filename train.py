from pytorch_lightning.utilities.cli import LightningCLI
import torch.backends

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER


torch.backends.cuda.max_split_size_mb = 64

# enable cudnn optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# force cuDNN to find a valid convolution algorithm
torch.backends.cudnn.enabled = True

def main():
    cli = LightningCLI(
        LitTAMER,
        HMEDatamodule,
        save_config_overwrite=True,
        trainer_defaults={
            'precision': 16,  # enable mixed precision
            'accumulate_grad_batches': 2,  # reduce memory spikes
            'auto_scale_batch_size': 'binsearch',  # dynamically reduce batch size
            # 'gpus': 1,
        }
    )


if __name__ == "__main__":
    main()