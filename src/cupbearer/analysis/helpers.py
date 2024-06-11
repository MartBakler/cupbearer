from dataclasses import dataclass
from typing import Any, Callable

from pathlib import Path
import torch
from datasets import concatenate_datasets
import tqdm
from einops import rearrange
from loguru import logger

from cupbearer import utils, scripts, detectors
from cupbearer.data import MixedData, HuggingfaceDataset
from cupbearer.detectors.activation_based import ActivationCache
from cupbearer.detectors.statistical.atp_detector import atp
from cupbearer.detectors.statistical.helpers import update_covariance
from cupbearer.tasks import Task
import gc
import pdb

class StatisticsCollector:
    # TODO: this is just copied from ActivationBasedDetector, should be refactored
    def __init__(
        self,
        activation_names: list[str],
        activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        cache: ActivationCache | None = None,
    ):
        self.activation_names = activation_names
        self.activation_processing_func = activation_processing_func
        self.cache = cache

    def set_model(self, model: torch.nn.Module):
        # This is separate from __init__ because we want to be able to set the model
        # automatically based on the task, instead of letting the user pass it in.
        # On the other hand, it's separate from train() because we might need to set
        # the model even when just using the detector for inference.
        #
        # Subclasses can implement more complex logic here.
        self.model = model

    def _get_activations_no_cache(self, inputs) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        inputs = utils.inputs_to_device(inputs, device)
        acts = utils.get_activations(self.model, self.activation_names, inputs)

        # Can be used to for example select activations at specific token positions
        if self.activation_processing_func is not None:
            acts = {
                k: self.activation_processing_func(v, inputs, k)
                for k, v in acts.items()
            }
        gc.collect()

        return acts

    def get_activations(self, batch) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)

        if self.cache is None:
            return self._get_activations_no_cache(inputs)

        return self.cache.get_activations(
            inputs, self.activation_names, self._get_activations_no_cache
        )

    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        self.between_class_variances = {
            k: torch.tensor(0.0, device=device) for k in self.activation_names
        }
        if any(len(size) != 1 for size in activation_sizes.values()):
            logger.debug(
                "Received multi-dimensional activations, will only learn "
                "covariances along last dimension and treat others independently. "
                "If this is unintentional, pass "
                "`activation_preprocessing_func=utils.flatten_last`."
            )
        self.means = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        self.normal_means = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        self.anomalous_means = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        # These are not the actual covariance matrices, they're missing a normalization
        # factor that we apply at the end.
        self._Cs = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._normal_Cs = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._anomalous_Cs = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._ns = {k: 0 for k in activation_sizes.keys()}
        self._normal_ns = {k: 0 for k in activation_sizes.keys()}
        self._anomalous_ns = {k: 0 for k in activation_sizes.keys()}

    def batch_update(self, activations: dict[str, torch.Tensor], labels: torch.Tensor):
        assert labels.ndim == 1
        labels = labels.bool()

        for k, activation in activations.items():
            # Flatten the activations to (batch, dim)
            normal_activation = rearrange(
                activation[~labels], "batch ... dim -> (batch ...) dim"
            )
            anomalous_activation = rearrange(
                activation[labels], "batch ... dim -> (batch ...) dim"
            )
            activation = rearrange(activation, "batch ... dim -> (batch ...) dim")

            # Update covariances and means
            self.means[k], self._Cs[k], self._ns[k] = update_covariance(
                self.means[k], self._Cs[k], self._ns[k], activation
            )

            if normal_activation.shape[0] > 0:
                (
                    self.normal_means[k],
                    self._normal_Cs[k],
                    self._normal_ns[k],
                ) = update_covariance(
                    self.normal_means[k],
                    self._normal_Cs[k],
                    self._normal_ns[k],
                    normal_activation,
                )

            if anomalous_activation.shape[0] > 0:
                (
                    self.anomalous_means[k],
                    self._anomalous_Cs[k],
                    self._anomalous_ns[k],
                ) = update_covariance(
                    self.anomalous_means[k],
                    self._anomalous_Cs[k],
                    self._anomalous_ns[k],
                    anomalous_activation,
                )

    def train(
        self,
        data: MixedData,
        *,
        batch_size: int = 1024,
        pbar: bool = True,
        max_steps: int | None = None,
    ):
        # Adapted from StatisticalDetector.train
        # TODO: figure out a way to refactor

        assert isinstance(data, MixedData)
        assert data.return_anomaly_labels

        with torch.inference_mode():
            data_loader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=True
            )
            example_batch, example_labels = next(iter(data_loader))
            example_activations = self.get_activations(example_batch)

            # v is an entire batch, v[0] are activations for a single input
            activation_sizes = {k: v[0].size() for k, v in example_activations.items()}
            self.init_variables(
                activation_sizes, device=next(iter(example_activations.values())).device
            )

            if pbar:
                data_loader = tqdm.tqdm(
                    data_loader, total=max_steps or len(data_loader)
                )

            for i, (batch, labels) in enumerate(data_loader):
                if max_steps and i >= max_steps:
                    break
                activations = self.get_activations(batch)
                self.batch_update(activations, labels)

        # Post processing for covariance
        with torch.inference_mode():
            self.covariances = {k: C / (self._ns[k] - 1) for k, C in self._Cs.items()}
            if any(torch.count_nonzero(C) == 0 for C in self.covariances.values()):
                raise RuntimeError("All zero covariance matrix detected.")

            self.normal_covariances = {
                k: C / (self._normal_ns[k] - 1) for k, C in self._normal_Cs.items()
            }
            self.anomalous_covariances = {
                k: C / (self._anomalous_ns[k] - 1)
                for k, C in self._anomalous_Cs.items()
            }

            self.total_variances = {
                k: self.covariances[k].trace() for k in self.activation_names
            }
            self.normal_variances = {
                k: self.normal_covariances[k].trace() for k in self.activation_names
            }
            self.anomalous_variances = {
                k: self.anomalous_covariances[k].trace() for k in self.activation_names
            }

            self.within_class_variances = {
                k: (
                    self._normal_ns[k] * self.normal_variances[k]
                    + self._anomalous_ns[k] * self.anomalous_variances[k]
                )
                / self._ns[k]
                for k in self.activation_names
            }

            self.between_class_variances = {
                k: self.total_variances[k] - self.within_class_variances[k]
                for k in self.activation_names
            }

class AttributionCollector(detectors.statistical.statistical.ActivationCovarianceBasedDetector):
    def __init__(
            self, 
            shapes: dict[str, tuple[int, ...]], 
            output_func: Callable[[torch.Tensor], torch.Tensor],
            ablation: str = 'zero',
            activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
            | None = None,
            n_pcs: int = 25
            ):

        activation_names = [k+'.output' for k in shapes.keys()]

        super().__init__(activation_names, activation_processing_func)
        self.shapes = shapes
        self.output_func = output_func
        self.ablation = ablation
        self.n_pcs = n_pcs

    def post_train(self):
        self.effects = self._effects

    def _get_trained_variables(self, saving: bool = False):
        return{
            "effects": self.effects,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self.effects = variables["effects"]
        self.noise = variables["noise"]

    def post_covariance_training(self, **kwargs):
        pass

    def layerwise_scores(self, batch):
        pass

    @torch.enable_grad()
    def train(
        self,
        trusted_data: torch.utils.data.Dataset,
        untrusted_data: torch.utils.data.Dataset | None,
        save_path: Path | str | None,
        batch_size: int = 1,
        n_samples: int | None = None,
        seed: int = 42,
        **kwargs,
    ):

        assert trusted_data is not None
        dtype = self.model.hf_model.dtype
        device = self.model.hf_model.device

        with torch.no_grad():
            self.noise = self.get_noise_tensor(trusted_data, batch_size, device, dtype)

        # Why shape[-2]? We are going to sum over the last dimension during attribution
        # patching. We'll then use the second-to-last dimension as our main dimension
        # to fit Gaussians to (all earlier dimensions will be summed out first).
        # This is kind of arbitrary and we're putting the onus on the user to make
        # sure this makes sense.
        self._n = 0
        if self.ablation == 'pcs':
            noise_batch_size = self.noise[list(self.noise.keys())[0]][0].shape[0]
        else:
            noise_batch_size = self.noise[list(self.noise.keys())[0]].shape[0]
        
        self._effects = {
            name: torch.zeros(min(n_samples, len(trusted_data)), noise_batch_size * 32, device=device)
            for name, shape in self.shapes.items()
        }

        labels = []

        torch.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(trusted_data, batch_size=batch_size, shuffle=True)
        n_datapoints = 0
        for i, (batch, label) in tqdm.tqdm(enumerate(dataloader)):
            inputs = utils.inputs_from_batch(batch)
            with atp(self.model, self.noise, head_dim=128) as effects:
                out = self.model(inputs).logits
                out = self.output_func(out)
                # assert out.shape == (batch_size,), out.shape
                out.backward()

            for name, effect in effects.items():
                # Get the effect at the last token
                effect = effect[:, :, -1]
                # Merge the last dimensions
                effect = effect.reshape(batch_size, -1)
                self._effects[name][i] = effect
            
            labels.append(label)

            n_datapoints += batch_size
            if n_samples is not None and n_datapoints >= n_samples:
                break

        self.effects = self._effects

        return torch.cat(labels)

    def get_noise_tensor(self, trusted_data, batch_size, device, dtype, 
                         subset_size=1000, activation_batch_size=16):
        if isinstance(trusted_data, MixedData):
            td = HuggingfaceDataset(concatenate_datasets([trusted_data.normal_data.hf_dataset, 
                                                            trusted_data.anomalous_data.hf_dataset]), 
                                                            text_key=trusted_data.normal_data.text_key, 
                                                            label_key=trusted_data.normal_data.label_key)
        else:
            td = trusted_data
        if self.ablation == 'mean':
            indices = torch.randperm(len(trusted_data))[:subset_size]
            subset = HuggingfaceDataset(
                td.hf_dataset.select(indices),
                text_key=td.text_key,
                label_key=td.label_key
            )

            super().train(subset, None, batch_size=activation_batch_size)
            return {k.replace('.output', ''): v.unsqueeze(0) for k, v in self.means.items()}

        elif self.ablation == 'pcs':
            super().train(td, None, batch_size=activation_batch_size)
            pcs = {}
            for k, C in self.covariances.items():
                eigenvalues, eigenvectors = torch.linalg.eigh(C)
                sorted_indices = eigenvalues.argsort(descending=True)
                principal_components = eigenvectors[:, sorted_indices[:self.n_pcs]]
                principal_components /= torch.norm(principal_components, dim=0)
                mean_activations = torch.matmul(principal_components.T, self._means[k])
                pcs[k.replace('.output', '')] = (principal_components.T, mean_activations)
            return pcs

        elif self.ablation == 'zero':
            return {
                name: torch.zeros((batch_size, 1, *shape), device=device, 
                                  dtype=dtype)
                for name, shape in self.shapes.items()
            }

@dataclass
class AttributionTaskData:
    effects: dict[str, torch.Tensor]
    labels: torch.Tensor
    detector: AttributionCollector

    @staticmethod
    def from_task(
        task: Task,
        shapes: dict[str, tuple[int, ...]], 
        ablation: str = 'mean',
        activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
            | None = None,
        batch_size: int = 1,
        n_samples: int = 1024,
        seed: int = 42
    ):

        no_token = task.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
        yes_token = task.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
        effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device="cpu")

        def effect_prob_func(logits):
            assert logits.ndim == 3
            probs = logits[:, -1, effect_tokens].diff(dim=1)

            return probs.sum()

        collector = AttributionCollector(
            shapes=shapes,
            output_func=effect_prob_func,
            ablation=ablation,
            activation_processing_func=activation_processing_func,
        )
        emb = task.model.hf_model.get_input_embeddings()
        emb.requires_grad_(True)

        collector.set_model(task.model)

        labels = collector.train(
            trusted_data=task.test_data,
            untrusted_data=task.untrusted_train_data,
            save_path=None,
            batch_size=batch_size,
            pbar=True,
            n_samples=n_samples,
            seed=seed
        )

        return AttributionTaskData(
            effects=collector.effects,
            labels=labels,
            detector=collector
        )

@dataclass
class TaskData:
    activations: dict[str, torch.Tensor]
    activations_train: dict[str, torch.Tensor]
    labels: torch.Tensor
    labels_train: torch.Tensor
    collector: StatisticsCollector

    @staticmethod
    def from_task(
        task: Task,
        activation_names: list[str],
        n_samples: int = 64,
        activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        cache: ActivationCache | None = None,
        batch_size: int = 4
    ):
        collector = StatisticsCollector(
            activation_names=activation_names,
            activation_processing_func=activation_processing_func,
            cache=cache,
        )
        collector.set_model(task.model)

        dataloader = torch.utils.data.DataLoader(
            task.test_data, batch_size=batch_size, shuffle=True
        )
        all_activations = []
        all_labels = []
        total_samples = 0
        for batch, test_labels in tqdm.tqdm(dataloader):
            activations = collector.get_activations(batch)
            all_activations.append({k: v.cpu() for k, v in activations.items()})
            all_labels.append(test_labels)

            batch_size = test_labels.size(0)
            total_samples += batch_size
            if total_samples > n_samples:
                break
        activations = {k: torch.cat([a[k] for a in all_activations]) for k in all_activations[0].keys()}
        test_labels = torch.cat(all_labels)

        task.untrusted_train_data.return_anomaly_labels = True

        dataloader_train = torch.utils.data.DataLoader(
            task.untrusted_train_data, batch_size=batch_size, shuffle=True
        )
        all_activations_train = []
        all_labels_train = []
        for batch, labels in tqdm.tqdm(dataloader_train):
            activations_train = collector.get_activations(batch)
            all_activations_train.append({k: v.cpu() for k, v in activations_train.items()})
            all_labels_train.append(labels)
        activations_train = {k: torch.cat([a[k] for a in all_activations_train]) for k in all_activations_train[0].keys()}
        labels_train = torch.cat(all_labels_train)
        # collector.train(data=task.test_data, batch_size=batch_size)
        return TaskData(
            activations=activations,
            activations_train=activations_train,
            labels=test_labels,
            labels_train=labels_train,
            collector=collector,
        )


def top_eigenvectors(matrix: torch.Tensor, n: int):
    mps = False
    if matrix.is_mps:
        mps = True
        matrix = matrix.cpu()
    eig = torch.linalg.eigh(matrix)
    eig_vectors = eig.eigenvectors[:, -n:]
    if mps:
        eig_vectors = eig_vectors.to("mps")
    return eig_vectors
