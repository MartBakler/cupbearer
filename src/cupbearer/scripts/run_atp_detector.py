import torch
import argparse
from cupbearer import detectors, tasks, utils, scripts
from cupbearer.detectors.statistical import atp_detector
from pathlib import Path
from cupbearer.detectors.activations import get_last_token_activation_function_for_task

def main(detector_type, first_layer, last_layer, model_name, features, ablation, k=20):
    layers = list(range(first_layer, last_layer + 1))

    task = tasks.quirky_lm(include_untrusted=True, mixture=True, standardize_template=True)

    no_token = task.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
    yes_token = task.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
    effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device="cpu")

    def effect_prob_func(logits):
        assert logits.ndim == 3
        probs = logits.softmax(-1)

        return probs[:, -1, effect_tokens].sum()

    if features == "attribution":
        batch_size = 1
        eval_batch_size = 1
        layer_dict = {f"hf_model.base_model.model.model.layers.{layer}.self_attn.output": (4096,) for layer in layers}

        if detector_type == "mahalonobis":
            detector = atp_detector.MahaAttributionDetector(layer_dict, effect_prob_func, ablation=ablation)
        elif detector_type == "isoforest":
            detector = atp_detector.IsoForestAttributionDetector(layer_dict, effect_prob_func, ablation=ablation)
        elif detector_type == "lof":
            detector = atp_detector.LOFAttributionDetector(layer_dict, effect_prob_func, k=k, ablation=ablation)

        emb = task.model.hf_model.get_input_embeddings()
        emb.requires_grad_(True)

    elif features == "activations":
        batch_size = 20
        eval_batch_size = 20


        layer_list = [f"hf_model.base_model.model.model.layers.{layer}.input_layernorm.input" for layer in layers]

        if detector_type == "mahalonobis":
            activation_processing_function = get_last_token_activation_function_for_task(task)
            detector = detectors.MahalanobisDetector(
                            activation_names=layer_list,
                            activation_processing_func=activation_processing_function,
                        )
        elif detector_type == "isoforest":
            raise NotImplementedError
        elif detector_type == "lof":
            activation_processing_function = get_last_token_activation_function_for_task(task)
            detector = detectors.statistical.lof_detector.LOFDetector(
                            activation_names=layer_list,
                            activation_processing_func=activation_processing_function,
                        )

    save_path = f"logs/quirky/sciq-{detector_type}-{features}-{model_name}-{first_layer}-{last_layer}-{args.ablation}"

    if Path(save_path).exists():
        detector.load_weights(Path(save_path) / "detector")
        scripts.eval_detector(task, detector, save_path, pbar=True, batch_size=eval_batch_size)
    else:
        scripts.train_detector(task, detector, 
                        batch_size=batch_size, 
                        save_path=save_path, 
                        eval_batch_size=eval_batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ATP Detector")
    parser.add_argument('--detector_type', type=str, required=True, help='Type of detector to use')
    parser.add_argument('--model_name', type=str, default='', help='Name of the detector to use')
    parser.add_argument('--first_layer', type=int, required=True, help='First layer to use')
    parser.add_argument('--last_layer', type=int, required=True, help='Last layer to use')
    parser.add_argument('--features', type=str, required=True, help='Features to use (attribution or activations)')
    parser.add_argument('--ablation', type=str, default='mean', help='Ablation to use (mean, zero)')

    args = parser.parse_args()
    main(args.detector_type, args.first_layer, args.last_layer, args.model_name, args.features, args.ablation)

