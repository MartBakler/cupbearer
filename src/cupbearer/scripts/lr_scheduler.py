from typing import Protocol, Union

import math
import torch
import torch.optim as optim


class LRSchedulerBuilder(Protocol):
    def __call__(
        self, optimizer: optim.Optimizer, **kwargs
    ) -> optim.lr_scheduler.LRScheduler:
        ...

Num = Union[int, float]

def step_lerp(start: torch.Tensor, end: torch.Tensor, step: Num, steps_to_end: Num):
    if step > steps_to_end:
        return end
    weight = step / steps_to_end
    return torch.lerp(start, end, weight)


def step_cosine_decay(start: torch.Tensor, end: torch.Tensor, step: Num, steps_to_end: Num):
    return torch.lerp(
        start,
        end,
        (1 - torch.cos(step_lerp(torch.tensor(0.0).to(start), torch.tensor(math.pi).to(start), step, steps_to_end)))
        / 2,
    )


def get_cosine_func_with_warmup(
    lr_warmup_steps: int, total_steps: int, lr: float, warmup_up_start_lr_mul: float = 0.1, final_lr_mul: float = 0.1
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

# lr scheduler from 
# https://github.com/redwoodresearch/Measurement-Tampering/blob/main/measurement_tampering/train_fsdp.py
# as a class for semantic hpyerparemter logging
class CosineWarmupScheduler(optim.lr_scheduler.LambdaLR):
    # con
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int, 
        total_steps: int, 
        warmup_up_start_lr_mul: float = 0.1, 
        final_lr_mul: float = 0.1,
        **kwargs
    ):
        lr_lambda = get_cosine_func_with_warmup(
            lr_warmup_steps=num_warmup_steps,
            total_steps=total_steps,
            lr=1.0, # not sure why this is called lr
            warmup_up_start_lr_mul=warmup_up_start_lr_mul,
            final_lr_mul=final_lr_mul
        )
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda, **kwargs)