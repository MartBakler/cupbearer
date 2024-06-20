from cupbearer import utils
from cupbearer.data import MixedData
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
import json
import sklearn.metrics
import numpy as np
from tqdm import tqdm
from loguru import logger
import pdb

def maybe_auc(y_true, y_scores):
    try:
        return sklearn.metrics.roc_auc_score(y_true, y_scores)
    except ValueError:
        return np.nan

def measure_accuracy(task, batch_size=32, pbar=True, save_path=None, histogram_percentile=95):
    def get_scores(batch):
        inputs = utils.inputs_from_batch(batch)
        encoding = task.model.tokenize(inputs)
        mask = encoding['attention_mask']

        no_token = task.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
        yes_token = task.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
        effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device=task.model.device)

        logits = task.model(inputs).logits[..., effect_tokens][range(len(inputs)), mask.sum(dim=1) - 1]
        return torch.nn.functional.softmax(logits, dim=1)[:, 1]

    dataset = task.test_data
    assert isinstance(dataset, MixedData), type(dataset)

    dataset.return_labels = ['answer', 'anomaly', 'agreement']

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # For some methods, such as adversarial abstractions, it might matter how
        # normal/anomalous data is distributed into batches. In that case, we want
        # to mix them by default.
        shuffle=True,
    )

    metrics = defaultdict(dict)
    if save_path is not None:
        model_name = Path(save_path).parts[-1]
    assert 0 < histogram_percentile <= 100

    if pbar:
        test_loader = tqdm(test_loader, desc="Evaluating", leave=False)

    scores = []
    anomalies = []
    agreement = []
    answers = []

    # It's important we don't use torch.inference_mode() here, since we want
    # to be able to override this in certain detectors using torch.enable_grad().
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, (new_labels, new_anomaly, new_agreements) = batch
            new_scores = get_scores(inputs).cpu().numpy()
            scores.append(new_scores)
            anomalies.append(new_anomaly)
            agreement.append(new_agreements)
            answers.append(new_labels)
    scores = np.concatenate(scores)
    anomalies = np.concatenate(anomalies).astype(bool)
    agreement = np.concatenate(agreement).astype(bool)
    answers = np.concatenate(answers).astype(bool)
    pdb.set_trace()

    auc_roc = maybe_auc(
        answers,
        scores,
    )
    ap = sklearn.metrics.average_precision_score(
        y_true=answers,
        y_score=scores,
    )
    logger.info(f"AUC_ROC: {auc_roc:.4f}")
    logger.info(f"AP: {ap:.4f}")
    metrics["AUC_ROC"] = auc_roc
    metrics["AP"] = ap

    auc_roc_agree_bob = maybe_auc(
        answers[agreement][anomalies[agreement]],  
        scores[agreement][anomalies[agreement]],
    )
    ap_agree_bob = sklearn.metrics.average_precision_score(
        y_true=answers[agreement][anomalies[agreement]],
        y_score=scores[agreement][anomalies[agreement]],
    )
    logger.info(f"AUC_ROC_AGREE_BOB: {auc_roc_agree_bob:.4f}")
    logger.info(f"AP_AGREE_BOB: {ap_agree_bob:.4f}")
    metrics["AUC_ROC_AGREE_BOB"] = auc_roc_agree_bob
    metrics["AP_AGREE_BOB"] = ap_agree_bob

    auc_roc_agree_alice = maybe_auc(
        answers[agreement][~anomalies[agreement]],
        scores[agreement][~anomalies[agreement]],
    )
    ap_agree_alice = sklearn.metrics.average_precision_score(
        y_true=answers[agreement][~anomalies[agreement]],
        y_score=scores[agreement][~anomalies[agreement]],
    )
    logger.info(f"AUC_ROC_AGREE_ALICE: {auc_roc_agree_alice:.4f}")
    logger.info(f"AP_AGREE_ALICE: {ap_agree_alice:.4f}")
    metrics["AUC_ROC_AGREE_ALICE"] = auc_roc_agree_alice
    metrics["AP_AGREE_ALICE"] = ap_agree_alice

    auc_roc_disagree_bob = maybe_auc(
        answers[~agreement][anomalies[~agreement]],
        scores[~agreement][anomalies[~agreement]],
    )
    ap_disagree_bob = sklearn.metrics.average_precision_score(
        y_true=answers[~agreement][anomalies[~agreement]],
        y_score=scores[~agreement][anomalies[~agreement]],
    )
    logger.info(f"AUC_ROC_DISAGREE_BOB: {auc_roc_disagree_bob:.4f}")
    logger.info(f"AP_DISAGREE_BOB: {ap_disagree_bob:.4f}")
    metrics["AUC_ROC_DISAGREE_BOB"] = auc_roc_disagree_bob
    metrics["AP_DISAGREE_BOB"] = ap_disagree_bob

    auc_roc_disagree_alice = maybe_auc(
        answers[~agreement][~anomalies[~agreement]],
        scores[~agreement][~anomalies[~agreement]],
    )
    ap_disagree_alice = sklearn.metrics.average_precision_score(
        y_true=answers[~agreement][~anomalies[~agreement]],
        y_score=scores[~agreement][~anomalies[~agreement]],
    )
    logger.info(f"AUC_ROC_DISAGREE_ALICE: {auc_roc_disagree_alice:.4f}")
    logger.info(f"AP_DISAGREE_ALICE: {ap_disagree_alice:.4f}")
    metrics["AUC_ROC_DISAGREE_ALICE"] = auc_roc_disagree_alice
    metrics["AP_DISAGREE_ALICE"] = ap_disagree_alice

    if not save_path:
        return metrics

    save_path = Path(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    # Everything from here is just saving metrics and creating figures
    # (which we skip if they aren't going to be saved anyway).
    with open(save_path / "eval.json", "w") as f:
        json.dump(metrics, f)

    return metrics