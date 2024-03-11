from typing import Callable, Optional

import torch

from cupbearer.detectors.finetuning.finetuning import FinetuningAnomalyDetector
from cupbearer.utils import inputs_from_batch


class FinetuningConfidenceAnomalyDetector(FinetuningAnomalyDetector):
    def __init__(
        self,
        *args,
        out_filter: Optional[Callable] = None,
        score_reduction: Optional[Callable] = None,
        **kwargs,
    ):
        """
        out_filter - applied to finetuned outputs - useful for eg. selecting label of
        interest in multilabel
        score_reduction - applied to computed scores - useful for e.g. averaging over
        logits in multilabel
        """
        self.out_filter = out_filter
        self.score_reduction = score_reduction
        super().__init__(*args, **kwargs)

    def scores(self, batch):
        # computes "confidence" score, either taking highest class logits
        # or absolute value of

        # TODO: check shapes, compatibility with multiclass, multilabel, etc
        inputs = inputs_from_batch(batch)
        finetuned_output: torch.Tensor = self.finetuned_model(inputs)
        if self.out_filter:
            finetuned_output = self.out_filter(finetuned_output)

        if self.classify_task == "multiclass":  # TODO: test
            # batch # n classes tensor
            scores = finetuned_output.max(dim=-1)
        else:  # multilabel or binary
            scores = finetuned_output.abs()

        if self.score_reduction:
            scores = self.score_reduction(scores)

        # automatically applies mean reduction if multiple scores per batch
        if len(scores.shape) != 1:
            assert len(scores.shape) == 2
            scores = scores.mean(dim=-1)

        return scores
