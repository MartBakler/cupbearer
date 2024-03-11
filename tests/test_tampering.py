import pytest
import torch
from cupbearer import data, detectors, models, tasks
from cupbearer.scripts import eval_classifier, train_classifier, train_detector

# Ignore warnings about num_workers
pytestmark = pytest.mark.filterwarnings(
    "ignore"
    ":The '[a-z]*_dataloader' does not have many workers which may be a bottleneck. "
    "Consider increasing the value of the `num_workers` argument` to "
    "`num_workers=[0-9]*` in the `DataLoader` to improve performance."
    ":UserWarning"
)


@pytest.fixture(scope="module")
def pythia():
    transformer, tokenizer, emb_dim, max_len = models.transformers_hf.load_transformer(
        "pythia-14m"
    )
    return models.TamperingPredictionTransformer(
        model=transformer,
        tokenizer=tokenizer,
        embed_dim=emb_dim,
        max_length=max_len,
        n_sensors=3,
    )


@pytest.fixture(scope="module")
def diamond():
    return torch.utils.data.Subset(data.TamperingDataset("diamonds"), range(10))


@pytest.fixture
def measurement_tampering_task(pythia, diamond):
    return tasks.measurement_tampering(
        model=pythia, train_data=diamond, test_data=diamond
    )


@pytest.fixture(scope="module")
def measurement_predictor_path(pythia, diamond, module_tmp_path):
    train_loader = torch.utils.data.DataLoader(diamond, batch_size=2)

    train_classifier(
        train_loader=train_loader,
        model=pythia,
        num_labels=4,
        task="multilabel",
        path=module_tmp_path,
        max_steps=1,
        logger=False,
        log_every_n_steps=3,
    )

    assert (module_tmp_path / "checkpoints" / "last.ckpt").is_file()

    return module_tmp_path


@pytest.mark.slow
def test_eval_classifier(pythia, diamond, measurement_predictor_path):
    models.load(pythia, measurement_predictor_path)

    eval_classifier(
        data=diamond,
        model=pythia,
        path=measurement_predictor_path,
        max_batches=1,
        batch_size=2,
    )

    assert (measurement_predictor_path / "eval.json").is_file()


@pytest.mark.slow
def test_train_finetune_confidence_detector(
    pythia, measurement_tampering_task, tmp_path
):
    train_detector(
        task=measurement_tampering_task,
        detector=detectors.FinetuningConfidenceAnomalyDetector(
            out_filter=lambda x: x[:, 3]  # filters for cummulative measurment probe
        ),
        num_labels=4,
        classify_task="multilabel",
        use_untrusted=True,
        save_path=tmp_path,
        batch_size=2,
        eval_batch_size=2,
        max_steps=1,
    )
    assert (tmp_path / "detector.pt").is_file()

    assert (tmp_path / "histogram.pdf").is_file()
    assert (tmp_path / "eval.json").is_file()
