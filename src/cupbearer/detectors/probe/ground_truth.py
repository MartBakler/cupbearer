import warnings
from pathlib import Path

import torch 
from torch.utils.data import ConcatDataset, DataLoader
import lightning as L

from cupbearer.scripts.classifier import ClassificationTask, Classifier

from .probe import ProbeAnomalyDetector

class GroundTruthProbeDetector(ProbeAnomalyDetector):

     def train(
        self,
        trusted_data,
        untrusted_data,
        save_path: Path | str,
        *,
        optim_conf,
        optim_builder,
        lr_scheduler_conf, 
        lr_scheduler_builder,
        batch_size: int = 64,
        **trainer_kwargs,
    ):
        if trusted_data is None:
            raise ValueError("Finetuning detector requires trusted training data.")
        classifier = Classifier(
            self.model,
            num_classes=1,
            task="binary",
            optim_conf=optim_conf,
            optim_builder=optim_builder,
            lr_scheduler_conf=lr_scheduler_conf, 
            lr_scheduler_builder=lr_scheduler_builder,
            save_hparams=False,
        )

        # Create a DataLoader for the clean dataset
        dataset = ConcatDataset([trusted_data, untrusted_data])
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
                train_dataloaders=train_loader,
            )