from lightning.pytorch.trainer import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from utils.metrics import log_loss
from utils.perturbation import Perturbation


class IntervalSampling(Callback):

    def __init__(self, perturbation: Perturbation = Perturbation(), every_n_epochs: int = 5):
        super().__init__()
        self.interval_epoch = every_n_epochs
        self.perturbation = perturbation

    def on_validation_epoch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule):
        if (trainer.current_epoch + 1) % self.interval_epoch == 0:
            for batch in trainer.val_dataloaders:
                truth = batch["target"].to(pl_module.device)
                sequence = self.perturbation(truth.clone())
                pred = pl_module.sample(y=sequence, steps=300)
                log_loss(pred=pred, truth=truth, stage="val", lightning_module=pl_module)

