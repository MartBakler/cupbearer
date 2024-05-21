import torch
from einops import rearrange

from cupbearer.detectors.statistical.helpers import local_outlier_factor
from cupbearer.detectors.statistical.statistical import StatisticalDetector

class LOFDetector(StatisticalDetector):
    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        for k, size in activation_sizes.items():
            if len(size) not in (1, 2):
                raise ValueError(
                    f"Activation size {size} of {k} is not supported. "
                    "Activations must be either 1D or 2D (in which case separate "
                    "covariance matrices are learned along the first dimension)."
                )
        self._activations = {
            k: torch.empty((0, size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
    
    def batch_update(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            self._activations[k] = torch.cat([self._activations[k], activation], dim=0)

    def train(self, trusted_data, untrusted_data, **kwargs):
        super().train(
            trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs
        )

        # Post process
        with torch.inference_mode():
            self.activations = self._activations


    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        batch_size = next(iter(activations.values())).shape[0]

        # Reshape activations to (batch, dim) for computing distances
        for k, activation in activations.items():
            if activation.ndim == 3:
                activation = rearrange(
                    activation, "batch independent dim -> (batch independent) dim"
                )
            assert activation.ndim == 2, activation.shape
            activations[k] = activation

        distances = local_outlier_factor(
            activations,
            self.activations,
        )

        for k, v in distances.items():
            # Unflatten distances so we can take the mean over the independent axis
            distances[k] = rearrange(
                v, "(batch independent) -> batch independent", batch=batch_size
            ).mean(dim=1)

        return distances

    def _get_trained_variables(self, saving: bool = False):
        return {
            "activations": self.activations,
        }

    def _set_trained_variables(self, variables):
        self.activations = variables["activations"]
