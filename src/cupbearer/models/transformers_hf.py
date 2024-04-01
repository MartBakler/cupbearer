from typing import TypedDict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.codegen.modeling_codegen import CodeGenForCausalLM
from transformers.models.codegen.tokenization_codegen_fast import CodeGenTokenizerFast
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from cupbearer.models import HookedModel


class TokenDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class TamperingTokenDict(TokenDict):
    sensor_inds: torch.Tensor


AUTO_MODELS = {
    "code-gen": "Salesforce/codegen-350M-mono",
}
NEOX_MODELS = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-14m": "EleutherAI/pythia-14m",
}

HF_MODELS = {**AUTO_MODELS, **NEOX_MODELS}


def load_transformer(
    name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, int, int]:
    # TODO (odk) add others, generalize (all models you can get from AutoModel)
    if name in AUTO_MODELS:
        checkpoint = AUTO_MODELS[name]
        model: CodeGenForCausalLM = AutoModelForCausalLM.from_pretrained(checkpoint)
        tokenizer: CodeGenTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
        transformer = model.transformer
        tokenizer.pad_token = tokenizer.eos_token
        emb_dim = transformer.embed_dim
        max_len = tokenizer.model_max_length
    elif name in NEOX_MODELS:
        checkpoint = NEOX_MODELS[name]
        model: GPTNeoXForCausalLM = AutoModelForCausalLM.from_pretrained(checkpoint)
        tokenizer: GPTNeoXTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
        transformer = model.gpt_neox
        tokenizer.pad_token = tokenizer.eos_token
        config: GPTNeoXConfig = transformer.config
        emb_dim = config.hidden_size
        max_len = config.max_position_embeddings
    else:
        raise ValueError(f"unsupported model {name}")

    return transformer, tokenizer, emb_dim, max_len


class TransformerBaseHF(HookedModel):
    def __init__(self, model: PreTrainedModel, embed_dim: int):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim

    def set_tokenizer(
        cls, tokenizer: PreTrainedTokenizerBase
    ) -> PreTrainedTokenizerBase:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def default_names(self) -> list[str]:
        return ["final_layer_embeddings"]

    def get_single_token(self, x):
        tokens: TokenDict = self.tokenizer(x)
        return tokens["input_ids"][0]

    def get_embeddings(self, tokens: TokenDict) -> torch.Tensor:
        b, s = tokens["input_ids"].shape
        out: BaseModelOutputWithPast = self.model(**tokens)
        embeddings = out.last_hidden_state
        assert embeddings.shape == (b, s, self.embed_dim), embeddings.shape
        return embeddings


# TODO: test
class ClassifierTransformerHF(TransformerBaseHF):
    def __init__(
        self,
        model: PreTrainedModel,
        embed_dim: int,
        pad_token_id: int,
        num_classes: int,
    ):
        super().__init__(model=model, embed_dim=embed_dim)
        self.pad_token_id = pad_token_id
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    # TODO: fix
    def forward(self, x: TokenDict):
        # get embeddings
        embeddings = self.get_embeddings(x)

        # take mean across non-padded dimensions
        mask = x["input_ids"] != self.pad_token_id
        mask = mask.unsqueeze(-1)
        assert mask.shape == x["input_ids"] + (1,)
        assert embeddings.shape == x["input_ids"] + (self.embed_dim,)
        embeddings = embeddings * mask
        embeddings = embeddings.sum(dim=1) / mask.sum(dim=1)
        self.store("last_hidden_state", embeddings)

        # compute logits
        logits = self.classifier(embeddings)
        return logits


class TamperingPredictionTransformer(TransformerBaseHF):
    # TODO: factor out token processing, create interface for using tokenizer in dataset
    def __init__(self, model: PreTrainedModel, embed_dim: int, n_sensors: int = 3):
        super().__init__(model=model, embed_dim=embed_dim)
        self.n_sensors = n_sensors
        self.n_probes = self.n_sensors + 1  # +1 for aggregate measurements

        self.probes = nn.ModuleList(
            [nn.Linear(self.embed_dim, 1) for _ in range(self.n_probes)]
        )

    @property
    def default_names(self) -> list[str]:
        return ["probe_embs"]

    def set_tokenizer(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> PreTrainedTokenizerBase:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def forward(self, x: TamperingTokenDict):
        b = x["input_ids"].shape[0]
        # get embeddings
        embeddings = self.get_embeddings({k: x[k] for k in TokenDict.__required_keys__})
        sensor_inds_reshaped = (
            x["sensor_inds"].unsqueeze(-1).expand(b, self.n_sensors, self.embed_dim)
        )
        sensor_embs = torch.gather(embeddings, dim=1, index=sensor_inds_reshaped)
        last_emb = embeddings[:, -1, :].unsqueeze(dim=1)
        probe_embs = torch.concat([sensor_embs, last_emb], axis=1)
        assert probe_embs.shape == (b, self.n_probes, self.embed_dim)
        self.store("probe_embs", probe_embs)
        # get logits
        logits = torch.concat(
            [
                probe(emb)
                for probe, emb in zip(self.probes, torch.split(probe_embs, 1, dim=1))
            ],
            dim=1,
        )
        logits = logits.squeeze(dim=-1)
        assert logits.shape == (b, self.n_probes)
        return logits
