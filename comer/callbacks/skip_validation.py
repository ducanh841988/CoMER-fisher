from typing import Optional

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.distributed import rank_zero_info


class SkipValidationOnHighTrainLoss(Callback):
    """Skip validation when epoch training loss is above a threshold.

    Uses ``trainer.limit_val_batches = 0`` for the current validation pass
    (PyTorch Lightning 1.4 compatible), then restores the original limit.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        monitor: str = "train_loss",
    ) -> None:
        self.threshold = threshold
        self.monitor = monitor
        self._original_limit_val_batches: Optional[int] = None
        self._skipped = False

    def _should_validate(self, trainer: Trainer, batch_idx: int) -> bool:
        """Mirror PL training loop: validation at interval steps or last batch."""
        if not trainer.enable_validation:
            return False
        if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch != 0:
            return False
        if trainer.is_last_batch:
            return True
        if trainer.val_check_batch == float("inf"):
            return False
        return (batch_idx + 1) % int(trainer.val_check_batch) == 0

    def _get_epoch_train_loss(self, trainer: Trainer, pl_module) -> Optional[float]:
        if hasattr(pl_module, "get_epoch_train_loss"):
            return pl_module.get_epoch_train_loss()

        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return None
        return float(metric)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self._should_validate(trainer, batch_idx):
            return

        train_loss = self._get_epoch_train_loss(trainer, pl_module)
        if train_loss is None:
            return

        if train_loss > self.threshold:
            self._original_limit_val_batches = trainer.limit_val_batches
            trainer.limit_val_batches = 0
            self._skipped = True
            rank_zero_info(
                "Skipping validation at epoch %d: %s=%.4f > %.4f",
                trainer.current_epoch,
                self.monitor,
                train_loss,
                self.threshold,
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        if not self._skipped:
            return
        trainer.limit_val_batches = self._original_limit_val_batches
        self._skipped = False
        self._original_limit_val_batches = None
