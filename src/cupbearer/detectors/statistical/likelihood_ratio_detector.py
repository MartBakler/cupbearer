import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

from cupbearer.detectors.statistical.helpers import update_covariance

from .statistical import ActivationCovarianceBasedDetector


class LikelihoodRatioDetector(ActivationCovarianceBasedDetector):
    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        super().init_variables(activation_sizes, device)
        self._means_untrusted = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        self._Cs_untrusted = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._ns_untrusted = {k: 0 for k in activation_sizes.keys()}

    def batch_update_untrusted(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            if activation.ndim == 3:
                activation = rearrange(
                    activation, "batch independent dim -> (batch independent) dim"
                )
            assert activation.ndim == 2, activation.shape
            self._means_untrusted[k], self._Cs_untrusted[k], self._ns_untrusted[k] = update_covariance(
                self._means_untrusted[k], self._Cs_untrusted[k], self._ns_untrusted[k], activation
            )

    def post_covariance_training(self, **kwargs):
        pass

    def train(self, trusted_data, untrusted_data, **kwargs):
        super().train(trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs)

        # Second pass for untrusted data
        with torch.inference_mode():
            data_loader = DataLoader(untrusted_data, batch_size=kwargs.get('batch_size', 1024), shuffle=False)
            for batch in tqdm(data_loader):
                activations = self.get_activations(batch)
                self.batch_update_untrusted(activations)

            self.means_untrusted = self._means_untrusted
            self.covariances_untrusted = {k: C / (self._ns_untrusted[k] - 1) for k, C in self._Cs_untrusted.items()}
            if any(torch.count_nonzero(C) == 0 for C in self.covariances_untrusted.values()):
                raise RuntimeError("All zero covariance matrix detected in untrusted data.")

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        scores = {}

        for k, activation in activations.items():
            if activation.ndim == 3:
                activation = rearrange(
                    activation, "batch independent dim -> (batch independent) dim"
                )
            assert activation.ndim == 2, activation.shape

            trusted_dist = MultivariateNormal(self.means[k], self.covariances[k])
            untrusted_dist = MultivariateNormal(self.means_untrusted[k], self.covariances_untrusted[k])

            log_prob_trusted = trusted_dist.log_prob(activation)
            log_prob_untrusted = untrusted_dist.log_prob(activation)

            scores[k] = log_prob_trusted - log_prob_untrusted

        return scores
