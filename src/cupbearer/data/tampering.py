from typing import Literal, get_args

import torch
from datasets import load_dataset

TAMPERING_DATSETS = {
    "diamonds": "redwoodresearch/diamonds-seed0",
    "text_props": "redwoodresearch/text_properties",
    "gen_stories": "redwoodresearch/generated_stories",
}


InfoNames = Literal["is_correct", "is_clean"]


class TamperingDataset(torch.utils.data.Dataset):
    info_name_idxs: dict[InfoNames, int] = {
        info_name: i for i, info_name in enumerate(get_args(InfoNames))
    }

    def __init__(self, name: str, train: bool = True):
        # TODO: allow for local loading / saving
        super().__init__()
        self.train = train
        self.name = name

        hf_name = (
            TAMPERING_DATSETS[self.name]
            if self.name in TAMPERING_DATSETS
            else self.name
        )
        split = "train" if self.train else "validation"
        self.dataset = load_dataset(hf_name, split=split)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (
            sample["text"],
            torch.tensor(
                [*sample["measurements"], all(sample["measurements"])],
                dtype=torch.float32,
            ),
            *[
                torch.tensor(sample[info_name])
                for info_name in self.info_name_idxs.keys()
            ],
        )

    def __len__(self):
        return len(self.dataset)
