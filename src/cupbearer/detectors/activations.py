import torch
from cupbearer.tasks import Task


def get_last_token_activation_function_for_task(task: Task, randperm: bool = False, proj_dim: int = 2**8):
    """Get the activation function for the last token of each input."""
    def get_activation_at_last_token(
        activation: torch.Tensor, 
        inputs: list[str], 
        name: str
    ):
        """Get the activations at the last token of each input."""
        # The activation should be (batch, sequence, residual dimension)
        assert activation.ndim == 3, activation.shape
        assert activation.shape[-1] == 1024, activation.shape # 4096 is mistral, 1024 is pythia 410
        batch_size = len(inputs)

        # Tokenize the inputs to know how many tokens there are. It's a bit unfortunate
        # that we're doing this twice (once here, once in the model), but not a huge deal.
        tokens = task.model.tokenize(inputs)
        last_non_padding_index = tokens["attention_mask"].sum(dim=1) - 1

        acts = activation[range(batch_size), last_non_padding_index, :]

        if randperm:
            indices = torch.randperm(acts.size(-1))[:proj_dim]
            acts = acts[:, indices]

        return acts
    
    return get_activation_at_last_token


def get_pca_reduction(data_loader, max_steps, detector):
    # initial implementation, disregard any batching for now
    training_activations = []
    for i, batch in enumerate(data_loader):
        if max_steps and i >= max_steps:
            break
        activations = detector.get_activations(batch)
        training_activations.append(activations)
    # make it into a single dictionary, where the value is a unified tensor
    training_activations = {
        k: torch.cat([a[k] for a in training_activations], dim=0)
        for k in training_activations[0].keys()
    }
    pca_compontents = {}
    variance_explained = 0.95
    # scale the activations to zero mean and unit variance
    for k, activation in training_activations.items():
        activation -= activation.mean(dim=0)
        activation /= activation.std(dim=0)
        # get the covariance matrix
        cov = torch.cov(activation.T) 
        # get the eigenvectors
        eigvals, eigvecs = torch.linalg.eig(cov)
        # move to cpu for sorting
        eigvals = eigvals.cpu().real
        eigvecs = eigvecs.cpu().real
        # sort the eigenvectors by the eigenvalues
        eigvals, indices = eigvals.sort(descending=True)
        eigvecs = eigvecs[:, indices]
        # get the number of components that explain the variance
        n_components = 0
        explained_variance = 0
        for i, val in enumerate(eigvals):
            explained_variance += val
            if explained_variance / eigvals.sum() > variance_explained:
                n_components = i
                break
        # currently a bit stupid to send to cpu and then back to cuda
        pca_compontents[k] = eigvecs[:, :n_components+1].to(activation.device)
    # delete training activations bc its large
    del training_activations
    return pca_compontents