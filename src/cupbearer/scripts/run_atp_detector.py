import torch
from cupbearer import detectors, tasks, utils, scripts
from cupbearer.detectors.statistical import atp_detector



def main():
    task = tasks.quirky_lm(include_untrusted=True, mixture=True, standardize_template=True)

    no_token = task.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
    yes_token = task.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
    effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device="cpu")

    def effect_prob_func(logits):
        assert logits.ndim == 3
        probs = logits.softmax(-1)

        return probs[:, -1, effect_tokens].sum()

    detector = atp_detector.MahaAttributionDetector(
        {"hf_model.base_model.model.model.layers.16.self_attn": (4096,)}, effect_prob_func, task
    )

    emb = task.model.hf_model.get_input_embeddings()
    emb.requires_grad_(True)
    scripts.train_detector(task, detector, 
                        batch_size = 1, 
                        save_path=f"logs/quirky/sciq-mahalonobis-attribution", 
                        eval_batch_size=1)

if __name__ == '__main__':
    main()