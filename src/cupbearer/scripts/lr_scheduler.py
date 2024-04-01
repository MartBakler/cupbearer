import math
from typing import Protocol, Union

import torch


class LRSchedulerBuilder(Protocol):
    def __call__(
        self, optim: torch.optim.Optimizer, **kwargs
    ) -> torch.optim.lr_scheduler.LRScheduler:
        ...


Num = Union[int, float]


def CosineWarmUpBuilder(
    optim: torch.optim.Optimizer,
    lr_warmup_steps: int,
    total_steps: int,
    lr: float,
    warmup_up_start_lr_mul: float = 0.1,
    final_lr_mul: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    builder_func = get_cosine_func_with_warmup(
        lr_warmup_steps=lr_warmup_steps,
        total_steps=total_steps,
        lr=lr,
        warmup_up_start_lr_mul=warmup_up_start_lr_mul,
        final_lr_mul=final_lr_mul,
    )
    return torch.optim.lr_scheduler.LambdaLR(optim, builder_func)


# NOTE by odk: from https://github.com/redwoodresearch/Measurement-Tampering/blob/main/measurement_tampering/train_fsdp.py
# which in turn got from some FSDP bert example
def step_lerp(start: torch.Tensor, end: torch.Tensor, step: Num, steps_to_end: Num):
    if step > steps_to_end:
        return end
    weight = step / steps_to_end
    return torch.lerp(start, end, weight)


def step_cosine_decay(
    start: torch.Tensor, end: torch.Tensor, step: Num, steps_to_end: Num
):
    return torch.lerp(
        start,
        end,
        (
            1
            - torch.cos(
                step_lerp(
                    torch.tensor(0.0).to(start),
                    torch.tensor(math.pi).to(start),
                    step,
                    steps_to_end,
                )
            )
        )
        / 2,
    )


def get_cosine_func_with_warmup(
    lr_warmup_steps: int,
    total_steps: int,
    lr: float,
    warmup_up_start_lr_mul: float = 0.1,
    final_lr_mul: float = 0.1,
):
    total_post_warmup_steps = total_steps - lr_warmup_steps
    warmup_up_start_lr = warmup_up_start_lr_mul * lr
    end_of_warmup_lr = lr
    final_lr = lr * final_lr_mul

    def run_func(step: int):
        is_warmup = step < lr_warmup_steps
        if is_warmup:
            return step_lerp(
                torch.tensor(warmup_up_start_lr),
                torch.tensor(end_of_warmup_lr),
                step,
                lr_warmup_steps,
            )
        else:
            real_step = step - lr_warmup_steps
            return step_cosine_decay(
                torch.tensor(lr),
                torch.tensor(final_lr),
                real_step,
                total_post_warmup_steps,
            )

    return lambda x: float(run_func(x))
