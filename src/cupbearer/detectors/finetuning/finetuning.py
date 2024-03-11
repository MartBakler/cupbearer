import copy
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.scripts._shared import ClassificationTask, Classifier


class FinetuningAnomalyDetector(AnomalyDetector, ABC):
    def set_model(self, model):
        super().set_model(model)
        # We might as well make a copy here already, since whether we'll train this
        # detector or load weights for inference, we'll need to copy in both cases.
        self.finetuned_model = copy.deepcopy(self.model)

        # setting here b/c set_model effectively serves as init
        self.classify_task: ClassificationTask | None = None

    def train(
        self,
        trusted_data,
        untrusted_data,
        save_path: Path | str,
        *,
        use_untrusted: bool = False,
        num_classes: int | None = None,
        num_labels: int | None = None,
        classify_task: ClassificationTask = "multiclass",
        lr: float = 1e-3,
        batch_size: int = 64,
        **trainer_kwargs,
    ):
        if trusted_data is None:
            raise ValueError("Finetuning detector requires trusted training data.")
        self.classify_task = classify_task
        classifier = Classifier(
            self.finetuned_model,
            num_classes=num_classes,
            num_labels=num_labels,
            task=classify_task,
            lr=lr,
            save_hparams=False,
        )

        # Create a DataLoader for the clean dataset
        dataset = trusted_data
        if use_untrusted:
            dataset = ConcatDataset([trusted_data, untrusted_data])
        clean_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Finetune the model on the clean dataset
        trainer = L.Trainer(default_root_dir=save_path, **trainer_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "You defined a `validation_step` but have no `val_dataloader`."
                    " Skipping val loop."
                ),
            )
            trainer.fit(
                model=classifier,
                train_dataloaders=clean_loader,
            )

    def layerwise_scores(self, batch):
        raise NotImplementedError(
            "Layerwise scores don't exist for finetuning detector"
        )

    @abstractmethod
    def scores(self, batch) -> torch.Tensor:
        "Computes anomaly scores for the given inputs"

    def _get_trained_variables(self, saving: bool = False):
        return self.finetuned_model.state_dict()

    def _set_trained_variables(self, variables):
        self.finetuned_model.load_state_dict(variables)
