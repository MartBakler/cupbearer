from typing import Optional

import torch


def concat_to_single_layer(activations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {'all': torch.cat([v for k, v in activations.items()], dim=1)}

def update_covariance(curr_mean, curr_C, curr_n, new_data):
    # Should be (batch, dim)
    assert new_data.ndim == 2

    new_n = len(new_data)
    total_n = curr_n + new_n

    new_mean = new_data.mean(dim=0)
    delta_mean = new_mean - curr_mean
    updated_mean = (curr_n * curr_mean + new_n * new_mean) / total_n

    delta_data = new_data - new_mean
    new_C = torch.einsum("bi,bj->ij", delta_data, delta_data)
    updated_C = (
        curr_C
        + new_C
        + curr_n * new_n / total_n * torch.einsum("i,j->ij", delta_mean, delta_mean)
    )

    return updated_mean, updated_C, total_n


def batch_covariance(batches):
    dim = batches[0].shape[1]
    mean = torch.zeros(dim)
    C = torch.zeros((dim, dim))
    n = 0

    for batch in batches:
        mean, C, n = update_covariance(mean, C, n, batch)

    return mean, C / (n - 1)  # Apply Bessel's correction for sample covariance


def mahalanobis(
    activations: dict[str, torch.Tensor],
    means: dict[str, torch.Tensor],
    inv_covariances: dict[str, torch.Tensor],
    inv_diag_covariances: Optional[dict[str, torch.Tensor]] = None,
    along_eigenvectors: bool = False,
):
    """Compute Simplified Relative Mahalanobis distances for a batch of activations.

    The Mahalanobis distance for each layer is computed,
    and the distances are then averaged over layers.

    Args:
        activations: Dictionary of activations for each layer,
            each element has shape (batch, dim)
        means: Dictionary of means for each layer, each element has shape (dim,)
        inv_covariances: Dictionary of inverse covariances for each layer,
            each element has shape (dim, dim)
        inv_diag_covariances: Dictionary of inverse diagonal covariances for each layer,
            each element has shape (dim,).
            If None, the usual Mahalanobis distance is computed instead of the
            simplified relative Mahalanobis distance.

    Returns:
        Dictionary of Mahalanobis distances for each layer,
        each element has shape (batch,).
    """
    distances: dict[str, torch.Tensor] = {}
    for k, activation in activations.items():
        batch_size = activation.shape[0]
        activation = activation.view(batch_size, -1)
        delta = activation - means[k]
        assert delta.ndim == 2 and delta.shape[0] == batch_size

        if along_eigenvectors:
            sorted_indices = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[sorted_indices]
            eigvecs = eigvecs[:, sorted_indices]

            # Compute per-eigenvector Mahalanobis distance
            per_eigenvector_distances = torch.einsum("bi,ij->bj", delta, eigvecs) / torch.sqrt(eigvals)
            
            distances[k] = per_eigenvector_distances
        else:
            # Compute unnormalized negative log likelihood under a Gaussian:
            distance = torch.einsum("bi,ij,bj->b", delta, inv_covariances[k], delta)
            # if inv_diag_covariances is not None:
            #     # distance -= torch.einsum("bi,i->b", delta**2, inv_diag_covariances[k])
            #     distance = torch.einsum("bi,i->b", delta**2, inv_diag_covariances[k])
            distances[k] = distance
    return distances

def mahalanobis_from_data(test_data, saved_data, relative=True):
    saved_means = dict()
    saved_inv_covs = dict()
    saved_inv_diag_covs = None
    if relative:
        saved_inv_diag_covs = dict()

    for k, v in saved_data.items():
        saved_means[k] = v.mean(dim=0)
        cov = torch.cov(v.T)
        saved_inv_covs[k] = torch.linalg.pinv(cov, 1.e-5)
        if relative:
            saved_inv_diag_covs[k] = torch.where(torch.diag(cov) > 1.e-5, 1 / torch.diag(cov), 0)

    return mahalanobis(test_data, saved_means, saved_inv_covs, saved_inv_diag_covs)

def quantum_entropy(
    whitened_activations: dict[str, torch.Tensor],
    alpha: float = 4,
    batch_covariance: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Quantum Entropy score per layer."""
    distances: dict[str, torch.Tensor] = {}
    for k, activation in whitened_activations.items():
        activation = activation.flatten(start_dim=1)

        # Compute QUE-score
        if batch_covariance is None:
            centered_batch = activation - activation.mean(dim=0, keepdim=True)
            batch_cov = centered_batch.mT @ centered_batch
        else:
            batch_cov = batch_covariance[k]

        batch_cov_norm = torch.linalg.eigvalsh(batch_cov).max()
        exp_factor = torch.matrix_exp(alpha * batch_cov / batch_cov_norm)

        distances[k] = torch.einsum(
            "bi,ij,jb->b",
            activation,
            exp_factor,
            activation.mT,
        )
    return distances


def local_outlier_factor(
        activations: dict[str, torch.Tensor],
        saved_activations: dict[str, torch.Tensor],
        k: int = 20
) -> dict[str, torch.Tensor]:
    """Local outlier factor per layer"""
    distances: dict[str, torch.Tensor] = {}

    for name, layer_activations in activations.items():
        batch_size = len(layer_activations)
        full_activations = torch.cat([layer_activations, saved_activations[name]], dim=0)

        epsilon = 0.0001
        # Calculate pairwise squared Euclidean distances
        test_dist = torch.cdist(full_activations, full_activations).fill_diagonal_(torch.inf) + epsilon
        test_distances, indices = test_dist.topk(k, largest=False)

        # Calculate reachability distances
        k_dists = test_distances[:, -1, None].expand_as(test_distances)
        lrd = torch.max(test_distances, k_dists).mean(dim=1).reciprocal()

        # Assert finite valuesk
        assert torch.isfinite(lrd).all()

        lrd_ratios = lrd[indices] / lrd[:, None]
        distances[name] = (lrd_ratios.sum(dim=1) / k)[:batch_size]

    return distances