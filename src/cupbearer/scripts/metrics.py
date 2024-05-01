from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleDict, ModuleList
from torchmetrics import BootStrapper, Metric
from torchmetrics.classification.auroc import BinaryAUROC
from torchmetrics.wrappers import MultioutputWrapper
from torchmetrics.wrappers.abstract import WrapperMetric

# TODO: add docstrings, standardize formatting


class WeightedBinaryAUROCBootStrapper(BootStrapper):
    def __init__(
        self,
        base_metric: BinaryAUROC,
        num_bootstraps: int = 10,
        pos_weight: float = 0.5,
        mean: bool = True,
        std: bool = True,
        quantile: Optional[Union[float, Tensor]] = None,
        raw: bool = False,
        out_dict: bool = True,
        **kwargs: Any,
    ) -> None:
        super(BootStrapper, self).__init__(**kwargs)

        self.base_metric = base_metric
        self.metrics = ModuleList(
            [deepcopy(base_metric) for _ in range(num_bootstraps)]
        )
        self.num_bootstraps = num_bootstraps
        self.pos_weight = pos_weight

        self.mean = mean
        self.std = std
        self.quantile = quantile
        self.raw = raw

        self.out_dict = out_dict

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the state of the base metric."""
        self.base_metric.update(preds, target)

    def compute(self) -> Dict[str, Tensor]:  # TODO:
        """Compute the bootstrapped metric values.

        samples entire dataset up to this point

        returns either a dict of tensors, which can contain the following keys:
        ``mean``, ``std``, ``quantile`` and ``raw`` depending on how the class was
        initialized, or a tensor with values only

        """
        # reset metrics
        for metric in self.metrics:
            metric.reset()
        # seperate positive and negative examples
        preds = torch.concatenate(self.base_metric.metric_state["preds"], dim=0)
        target = torch.concatenate(self.base_metric.metric_state["target"], dim=0)
        pos_preds = preds[target == 1]
        neg_preds = preds[target == 0]
        n_pos_preds = pos_preds.shape[0]
        n_neg_preds = neg_preds.shape[0]
        # compute number of positive and negative for each sample
        n_samples = preds.shape[0]
        n_pos = int(n_samples * self.pos_weight)
        n_neg = int(n_samples * (1 - self.pos_weight))
        # for each boostrap metric
        for i in range(self.num_bootstraps):
            pos_idxs = (
                np.random.choice(n_pos_preds, size=n_pos) if n_pos_preds > 0 else []
            )
            neg_idxs = np.random.choice(n_neg_preds, n_neg) if n_neg_preds > 0 else []
            self.metrics[i](
                preds[[*pos_idxs, *neg_idxs]], target[[*pos_idxs, *neg_idxs]]
            )
        # resample
        out_dict = super().compute()
        if self.out_dict:
            return out_dict
        return torch.stack([v for v in out_dict.values()])


class ScalarMetric(Metric):
    val: Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("vals", default=torch.empty(()), dist_reduce_fx="cat")

    def update(self, val: Tensor):
        self.val = val

    def compute(self):
        return self.val


class DictOutWrapper(WrapperMetric):
    def __init__(
        self,
        base_metric: Metric,
        keys: list[str],
        out_metric: Metric = ScalarMetric(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_metric = base_metric
        self.metrics = ModuleDict({k: deepcopy(out_metric) for k in keys})

    def update(self, *args, **kwargs):
        self.base_metric.update(*args, **kwargs)

    def compute(self):
        # compute out and update metrics dict
        out = self.base_metric.compute()
        assert isinstance(out, dict)
        for k, val in out.items():
            self.metrics[k](val)
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the original forward method of the base metric class."""
        return super(WrapperMetric, self).forward(*args, **kwargs)


class MultioutDictWrapper(MultioutputWrapper):
    def _stack_dicts(self, outs: dict):
        return {k: torch.stack([o[k] for o in outs], 0) for k in outs[0].keys()}

    def compute(self):
        outs = [m.compute() for m in self.metrics]
        if isinstance(outs[0], dict):
            return self._stack_dicts(outs)
        return torch.stack(outs, 0)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call underlying forward methods and aggregate the results if they're non-null

        We override this method to ensure that state variables get copied over on the
        underlying metrics.

        """
        reshaped_args_kwargs = self._get_args_kwargs_by_output(*args, **kwargs)
        results = [
            metric(*selected_args, **selected_kwargs)
            for metric, (selected_args, selected_kwargs) in zip(
                self.metrics, reshaped_args_kwargs
            )
        ]
        if results[0] is None:
            return None
        if isinstance(results[0], dict):
            return self._stack_dicts(results)
        return torch.stack(results, 0)
