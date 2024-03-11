import torch
import torch.nn.functional as F

from cupbearer.detectors.finetuning.finetuning import FinetuningAnomalyDetector
from cupbearer.utils import inputs_from_batch


class FinetuningShiftAnomalyDetector(FinetuningAnomalyDetector):
    def scores(self, batch):
        inputs = inputs_from_batch(batch)
        original_output = self.model(inputs)
        finetuned_output = self.finetuned_model(inputs)

        # F.kl_div requires log probabilities for the input, normal probabilities
        # are fine for the target.
        log_finetuned_p = finetuned_output.log_softmax(dim=-1)
        original_p = original_output.softmax(dim=-1)

        # This computes KL(original || finetuned), the argument order for the pytorch
        # function is swapped compared to the mathematical notation.
        # See https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # This is the same direction of KL divergence that Redwood used in one of their
        # projects, though I don't know if they had a strong reason for it.
        # Arguably a symmetric metric would make more sense, but might not matter much.
        #
        # Also note we don't want pytorch to do any reduction, since we want to
        # return individual scores for each sample.
        kl = F.kl_div(log_finetuned_p, original_p, reduction="none").sum(-1)

        if torch.any(torch.isinf(kl)):
            # We'd get an error anyway once we compute eval metrics, but better to give
            # a more specific one here.
            raise ValueError("Infinite KL divergence")

        return kl
