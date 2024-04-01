from typing import Protocol

import torch.optim as optim


class LRSchedulerBuilder(Protocol):
    def __call__(
        self, optimizer: optim.Optimizer, **kwargs
    ) -> optim.lr_scheduler.LRScheduler:
        ...
