# from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

cli = LightningCLI(
    LitTAMER,
    HMEDatamodule,
    trainer_class=Trainer,
    save_config_overwrite=True,
    # trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=False)},
)
