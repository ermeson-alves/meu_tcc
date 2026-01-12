
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SaveLastOrOnException(Callback):
    def __init__(self, dirpath: str, prefix: str = "locca", weights_only: bool = False):
        self.dirpath = Path(dirpath)
        self.prefix = prefix
        self.weights_only = weights_only

    def _ckpt_path(self, trainer: pl.Trainer, pl_module: pl.LightningModule, tag: str) -> str:
        phase = getattr(pl_module, "phase", "na")
        repet = getattr(pl_module, "repetition", "na")
        kfold = getattr(pl_module, "kfolditer", "na")

        epoch = trainer.current_epoch
        step = trainer.global_step
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.dirpath.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{self.prefix}"
            f"_phase{phase}"
            f"_repet{repet}"
            f"_kfolditer{kfold}"
            f"_epoch{epoch:03d}"
            f"_step{step}"
            f"_{tag}"
            f"_{ts}.ckpt"
        )
        return str(self.dirpath / filename)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if getattr(trainer, "is_global_zero", True):  # seguro em DDP
            path = self._ckpt_path(trainer, pl_module, tag="last")
            trainer.save_checkpoint(path)  # salva completo (modelo+optimizer+etc)

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException):
        if getattr(trainer, "is_global_zero", True):
            tag = f"exception-{type(exception).__name__}"
            path = self._ckpt_path(trainer, pl_module, tag=tag)

            # compatibilidade: nem toda versão tem weights_only em save_checkpoint
            try:
                trainer.save_checkpoint(path, weights_only=self.weights_only)
            except TypeError:
                trainer.save_checkpoint(path)
