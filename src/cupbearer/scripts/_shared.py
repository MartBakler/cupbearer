import math
from dataclasses import dataclass, asdict
from typing import TypedDict

import lightning as L
import torch
from torchmetrics.classification import Accuracy
from typing_extensions import Literal, Union

from cupbearer.models import HookedModel

ClassificationTask = Literal["binary", "multiclass", "multilabel"]

#TODO (odk) review this design with team
@dataclass
class LRSchedulerConf():
    warmup_steps: int = 0
    cosine_annealing: bool = False
    total_steps: int = 0
    warmup_start_lr_mul: float = 0.1
    final_lr_mul: float = 0.1


class Classifier(L.LightningModule):
    def __init__(
        self,
        model: HookedModel,
        lr: float,
        lr_scheduler_conf: dict = asdict(LRSchedulerConf()), 
        num_classes: int | None = None,
        num_labels: int | None = None,
        val_loader_names: list[str] | None = None,
        test_loader_names: list[str] | None = None,
        save_hparams: bool = True,
        task: ClassificationTask = "multiclass",
    ):
        super().__init__()
        if save_hparams:
            self.save_hyperparameters(ignore=["model"])
        if val_loader_names is None:
            val_loader_names = []
        if test_loader_names is None:
            test_loader_names = []

        self.model = model
        self.lr = lr
        self.lr_scheduler_conf = LRSchedulerConf(**lr_scheduler_conf)
        self.val_loader_names = val_loader_names
        self.test_loader_names = test_loader_names
        self.task = task
        self.loss_func = self._get_loss_func(self.task)
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

    def _get_loss_func(self, task):
        if task == "multiclass":
            return torch.nn.functional.cross_entropy
        return torch.nn.functional.binary_cross_entropy_with_logits

    def _shared_step(self, batch):
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, *_info = batch
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

    def on_validation_epoch_end(self):
        for i, name in enumerate(self.val_loader_names):
            self.log(f"{name}/acc_epoch", self.val_accuracy[i])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # construct lr scheduler
        if self.lr_scheduler_conf.cosine_annealing:
            schedule_func = get_cosine_func_with_warmup(
                self.lr_scheduler_conf.warmup_steps, 
                self.lr_scheduler_conf.total_steps, 
                self.lr, 
                self.lr_scheduler_conf.warmup_start_lr_mul,
                self.lr_scheduler_conf.final_lr_mul)
        elif self.lr_scheduler_conf["warmup_steps"] > 0:
            schedule_func = get_warmup_func(
                self.lr_scheduler_conf.warmup_steps,
                self.lr_scheduler_conf.total_steps,
                self.lr, 
                self.lr_scheduler_conf.warmup_start_lr_mul
            )
        else:
            return optim # no scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=schedule_func)
        lr_schedule_config = {
            "scheduler": scheduler,
            "interval": "step", # applies at every step rather than every epoch
        }
        return {
            "optimizer": optim,
            "lr_scheduler": lr_schedule_config
        }

# learning rate schedulers (TODO (odk) refactor somehow)
Num = Union[int, float]
def step_lerp(
    start: torch.Tensor, 
    end: torch.Tensor, 
    step: Num, 
    steps_to_end: Num
):
    if step > steps_to_end:
        return end
    weight = step / steps_to_end
    return torch.lerp(start, end, weight)


def step_cosine_decay(
    start: torch.Tensor, 
    end: torch.Tensor, 
    step: Num, 
    steps_to_end: Num
):
    return torch.lerp(
        start,
        end,
        (1 - torch.cos(step_lerp(torch.tensor(0.0).to(start), torch.tensor(math.pi).to(start), step, steps_to_end)))
        / 2,
    )

def get_warmup_func(
    lr_warmup_steps: int, 
    lr: float, 
    warmup_start_lr_mul: float = 0.1
):
    warmup_up_start_lr = warmup_start_lr_mul * lr
    end_of_warmup_lr = lr
    def warmup_func(step: int):
        return step_lerp(torch.tensor(warmup_up_start_lr), torch.tensor(end_of_warmup_lr), step, lr_warmup_steps)
    return lambda x: float(warmup_func(x))


def get_cosine_func_with_warmup(
    lr_warmup_steps: int, 
    total_steps: int, 
    lr: float, 
    warmup_up_start_lr_mul: float = 0.1, 
    final_lr_mul: float = 0.1
):
    total_post_warmup_steps = total_steps - lr_warmup_steps
    warmup_up_start_lr = warmup_up_start_lr_mul * lr
    end_of_warmup_lr = lr
    final_lr = lr * final_lr_mul

    def run_func(step: int):
        is_warmup = step < lr_warmup_steps
        if is_warmup:
            return step_lerp(torch.tensor(warmup_up_start_lr), torch.tensor(end_of_warmup_lr), step, lr_warmup_steps)
        else:
            real_step = step - lr_warmup_steps
            return step_cosine_decay(torch.tensor(lr), torch.tensor(final_lr), real_step, total_post_warmup_steps)

    return lambda x: float(run_func(x))