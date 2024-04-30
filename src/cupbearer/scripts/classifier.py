import lightning as L
import torch
from torchmetrics.classification import AUROC, Accuracy
from typing_extensions import Any, Callable, Literal, Union

from cupbearer.models import HookedModel

from .lr_scheduler import LRSchedulerBuilder

ClassificationTask = Literal["binary", "multiclass", "multilabel"]
Info = Union[torch.Tensor, dict[str, torch.Tensor]]


class Classifier(L.LightningModule):
    def __init__(
        self,
        model: HookedModel,
        optim_conf: dict = {},
        optim_builder: Callable[[Any], torch.optim.Optimizer] = torch.optim.Adam,
        lr_scheduler_conf: dict = {},
        lr_scheduler_builder: LRSchedulerBuilder | None = None,
        num_classes: int | None = None,
        num_labels: int | None = None,
        val_loader_names: list[str] | None = None,
        test_loader_names: list[str] | None = None,
        save_hparams: bool = True,
        task: ClassificationTask = "multiclass",
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        y_func: Callable[[torch.Tensor, Info], torch.Tensor] | None = None,
    ):
        super().__init__()
        if save_hparams:
            self.save_hyperparameters(ignore=["model"])
        if val_loader_names is None:
            val_loader_names = []
        if test_loader_names is None:
            test_loader_names = []

        self.model = model
        self.optim_conf = optim_conf
        self.optim_builder = optim_builder
        self.lr_scheduler_conf = lr_scheduler_conf
        self.lr_scheduler_builder = lr_scheduler_builder
        self.val_loader_names = val_loader_names
        self.test_loader_names = test_loader_names
        self.task = task
        self.loss_func = (
            loss_func if loss_func is not None else self._get_loss_func(self.task)
        )
        self.y_func = y_func
        self.train_accuracy = Accuracy(
            task=self.task, num_classes=num_classes, num_labels=num_labels
        )
        self.val_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task=self.task, num_classes=num_classes, num_labels=num_labels)
                for _ in val_loader_names
            ]
        )
        self.test_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task=self.task, num_classes=num_classes, num_labels=num_labels)
                for _ in test_loader_names
            ]
        )
        self.test_auroc = torch.nn.ModuleList(
            [
                AUROC(task=task, num_classes=num_classes, num_labels=num_labels, average=None)
                for _ in test_loader_names
            ]
        )

    def _get_loss_func(self, task):
        if task == "multiclass":
            return torch.nn.functional.cross_entropy
        return torch.nn.functional.binary_cross_entropy_with_logits

    def _shared_step(self, batch):
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
            info = batch.get("info", {})
        else:
            x, y, info = batch
        if self.y_func:
            y = self.y_func(y, info)
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.train_accuracy(logits, y)
        self.log("train/acc_step", self.train_accuracy)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y = self._shared_step(batch)
        name = self.test_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.test_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.test_accuracy[dataloader_idx])
        self.test_auroc[dataloader_idx](logits, y.to(torch.int))
        for i, val in enumerate(self.test_auroc[dataloader_idx].compute()):
            self.log(f"{name}/auroc_{i}_step", val)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y = self._shared_step(batch)
        name = self.val_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.val_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.val_accuracy[dataloader_idx])

    def on_train_epoch_end(self):
        self.log("train/acc_epoch", self.train_accuracy)

    def on_test_epoch_end(self):
        for i, name in enumerate(self.test_loader_names):
            self.log(f"{name}/acc_epoch", self.test_accuracy[i])
            for j, val in enumerate(self.test_auroc[i].compute()):
                self.log(f"{name}/auroc_{j}_epoch", val)

    def on_validation_epoch_end(self):
        for i, name in enumerate(self.val_loader_names):
            self.log(f"{name}/acc_epoch", self.val_accuracy[i])

    def configure_optimizers(self):
        optim = self.optim_builder(self.parameters(), **self.optim_conf)
        if not self.lr_scheduler_builder:
            return optim
        # build lr sceduler
        lr_scheduler = self.lr_scheduler_builder(
            optimizer=optim, **self.lr_scheduler_conf
        )
        lr_schedule_config = {
            "scheduler": lr_scheduler,
            "interval": "step",  # applies at every step rather than every epoch
        }
        return {"optimizer": optim, "lr_scheduler": lr_schedule_config}
