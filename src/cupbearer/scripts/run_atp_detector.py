import torch
import argparse
from cupbearer import detectors, tasks, utils, scripts
from cupbearer.detectors.statistical import atp_detector
from pathlib import Path
from cupbearer.detectors.activations import get_last_token_activation_function_for_task
from cupbearer.detectors.statistical.probe_detector import probe_error
from cupbearer.detectors.statistical.helpers import mahalanobis_from_data, local_outlier_factor
from cupbearer.scripts.measure_accuracy import measure_accuracy
import gc

datasets = [
    "capitals",
    "hemisphere",
    "population",
    "sciq",
    "sentiment",
    "nli",
    "authors",
    "addition",
    "subtraction",
    "multiplication",
    "modularaddition",
    # "squaring", Not trained yet
]


def main(dataset, detector_type, first_layer, last_layer, model_name, features, ablation, k=20, random_names=True, layerwise=False, alpha=8,  pca_basis=False):

    interval = max(1, (last_layer - first_layer) // 4)
    layers = list(range(first_layer, last_layer + 1, interval))

    task = tasks.quirky_lm(include_untrusted=True, mixture=True, standardize_template=True, dataset=dataset, random_names=random_names)

    no_token = task.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
    yes_token = task.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
    effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device="cpu")

    def effect_prob_func(logits):
        assert logits.ndim == 3
        probs = logits.softmax(-1)

        return probs[:, -1, effect_tokens].diff(1).sum()

    activation_processing_function = get_last_token_activation_function_for_task(task)

    answer_accuracy = False

    if detector_type == "accuracy":
        measure_accuracy(task, batch_size=32, pbar=False, save_path=f"logs/quirky/{dataset}-accuracy", histogram_percentile=95)
        return

    elif features == "attribution":
        batch_size = 1
        eval_batch_size = 1
        layer_dict = {f"hf_model.model.layers.{layer}.self_attn": (4096,) for layer in layers}

        if detector_type == "mahalanobis":
            detector = atp_detector.MahaAttributionDetector(
                layer_dict, 
                effect_prob_func, 
                ablation=ablation,
                activation_processing_func=activation_processing_function
                )
        elif detector_type == "isoforest":
            detector = atp_detector.IsoForestAttributionDetector(
                layer_dict, 
                effect_prob_func, 
                ablation=ablation, 
                activation_processing_func=activation_processing_function
            )
        elif detector_type == "lof":
            detector = atp_detector.LOFAttributionDetector(
                layer_dict, 
                effect_prob_func, 
                k=k, 
                ablation=ablation, 
                activation_processing_func=activation_processing_function
            )
        elif detector_type == 'que':
            detector = atp_detector.QueAttributionDetector(
                layer_dict, 
                effect_prob_func, 
                ablation=ablation, 
                activation_processing_func=activation_processing_function
            )

        emb = task.model.hf_model.get_input_embeddings()
        emb.requires_grad_(True)

    elif features == "activations":
        batch_size = 20
        eval_batch_size = 20


        layer_list = [f"hf_model.model.layers.{layer}.input_layernorm.input" for layer in layers]
        if 32 in layers:
            layer_list[-1] = "hf_model.model.norm.input"

        if detector_type == "mahalanobis":
            detector = detectors.MahalanobisDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
                pca_basis = pca_basis
            )
        elif detector_type == "isoforest":
            raise NotImplementedError
        elif detector_type == "lof":
            detector = detectors.statistical.lof_detector.LOFDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
            )
        elif detector_type == 'que':
            detector = detectors.statistical.que_detector.QuantumEntropyDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
                alpha=alpha
            )
        elif detector_type == 'spectral':
            detector = detectors.statistical.spectral_detector.SpectralSignatureDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
            )
        elif detector_type == 'likelihood':
            detector = detectors.statistical.likelihood_ratio_detector.LikelihoodRatioDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
            )
        elif detector_type == 'em':
            detector = detectors.statistical.likelihood_ratio_detector.ExpectationMaximisationDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
            )
        elif detector_type == 'probe_trajectory':
            detector = detectors.statistical.contrast_detector.ProbeTrajectoryDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
            )

    elif features == "trajectories":
        
        seq_len = 1 # Examine the trajectory for the last seq_len tokens

        batch_size = 8
        eval_batch_size = 8

        if detector_type == "mahalanobis":
            detector = detectors.statistical.trajectory_detector.MahaTrajectoryDetector(
                layers,
                seq_len = seq_len
            )
        elif detector_type == "lof":
            detector = detectors.statistical.trajectory_detector.LOFTrajectoryDetector(
                layers,
                seq_len = seq_len
            )

    elif features == "probe":
 
        layer_dict = {f"hf_model.model.layers.{layer}.self_attn": (4096,) for layer in layers}

        batch_size = 1
        eval_batch_size = 1

        if detector_type == "mahalanobis":
            detector = detectors.statistical.probe_detector.AtPProbeDetector(
            layer_dict,
            seq_len = 1,
            activation_processing_func=activation_processing_function,
            distance_function=mahalanobis_from_data,
            ablation=ablation
        )
        elif detector_type == "lof":
            detector = detectors.statistical.probe_detector.AtPProbeDetector(
            layer_dict,
            seq_len = 1,
            activation_processing_func=activation_processing_function,
            distance_function=local_outlier_factor,
            ablation=ablation
        )
        elif detector_type == "probe":
            detector = detectors.statistical.probe_detector.AtPProbeDetector(
            layer_dict,
            seq_len = 1,
            activation_processing_func=activation_processing_function,
            distance_function=probe_error,
            ablation=ablation
        )
 
        emb = task.model.hf_model.get_input_embeddings()
        emb.requires_grad_(True)

    elif features == "misconception-contrast":

        batch_size = 4
        eval_batch_size = 4

        layer_list = [f"hf_model.model.layers.{layer}.input_layernorm.input" for layer in layers]

        detector = detectors.statistical.MisconceptionContrastDetector(
            layer_list,
            activation_processing_func=activation_processing_function,
        )
    
    elif features == "iterative-rephrase":
        batch_size = 32
        eval_batch_size = 32

        detector = detectors.IterativeAnomalyDetector()

    save_path = f"logs/quirky/{dataset}-{detector_type}-{features}-{model_name}-{first_layer}-{last_layer}-{args.ablation}"

    if detector_type == "lof":
        save_path += f"-{k}"
    if detector_type == "que":
        save_path += f"-{alpha}"

    if Path(save_path).exists():
        detector.load_weights(Path(save_path) / "detector")
        scripts.eval_detector(task, detector, save_path, pbar=True, batch_size=eval_batch_size, train_from_test=False, layerwise=layerwise, answer_accuracy=answer_accuracy)
    else:
        scripts.train_detector(task, detector, 
                        batch_size=batch_size, 
                        save_path=save_path, 
                        eval_batch_size=eval_batch_size,
                        pbar=True,
                        train_from_test=False,
                        layerwise=layerwise,
                        answer_accuracy=answer_accuracy)
    
    del task, detector
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ATP Detector")
    parser.add_argument('--detector_type', type=str, required=True, help='Type of detector to use')
    parser.add_argument('--model_name', type=str, default='', help='Name of the detector to use')
    parser.add_argument('--first_layer', type=int, required=True, help='First layer to use')
    parser.add_argument('--last_layer', type=int, required=True, help='Last layer to use')
    parser.add_argument('--features', type=str, required=True, help='Features to use (attribution, trajectories, probe or activations)')
    parser.add_argument('--ablation', type=str, default='mean', help='Ablation to use (mean, zero, pcs)')
    parser.add_argument('--dataset', type=str, default='sciq', help='Dataset to use (sciq, addition)')
    parser.add_argument('--k', type=int, default=20, help='k to use for LOF')
    parser.add_argument('--sweep_layers', action='store_true', default=False, help='Sweep layers one by one')
    parser.add_argument('--alpha', type=float, default=4, help='Alpha to use for QUE')
    parser.add_argument('--sweep_alpha', action='store_true', default=False, help='Sweep alpha one by one')
    parser.add_argument('--layerwise', action='store_true', default=False, help='Evaluate layerwise instead of aggregated')
    parser.add_argument('--nonrandom_names', action='store_true', default=False, help='Avoid randomising names')
    parser.add_argument('--pca_basis', default=True, help='Apply PCA basis to activations')

    args = parser.parse_args()

    def run_main_with_args(dataset, first_layer, last_layer, alpha):
        main(
            dataset, 
            args.detector_type, 
            first_layer, 
            last_layer, 
            args.model_name, 
            args.features, 
            args.ablation, 
            k=args.k, 
            alpha=alpha, 
            layerwise=args.layerwise, 
            random_names=not args.nonrandom_names,
            pca_basis=args.pca_basis
        )

    if args.sweep_alpha:
        for alpha in range(0, 40, 4):
            if args.dataset == "all":
                for dataset in datasets:
                    run_main_with_args(dataset, args.first_layer, args.last_layer, alpha)
            else:
                run_main_with_args(args.dataset, args.first_layer, args.last_layer, alpha)
    elif args.sweep_layers:
        for layer in range(args.first_layer, args.last_layer + 1):
            main(args.dataset, args.detector_type, layer, layer, args.model_name, args.features, args.ablation, k=args.k, pca_basis=args.pca_basis)
    elif args.dataset == "all":
        for dataset in datasets:
            run_main_with_args(dataset, args.first_layer, args.last_layer, args.alpha, args.pca_basis)
    else:
        run_main_with_args(args.dataset, args.first_layer, args.last_layer, args.alpha, args.pca_basis)

