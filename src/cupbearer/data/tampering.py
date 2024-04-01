import torch
from datasets import load_dataset

TAMPERING_DATSETS = {
    "diamonds": "redwoodresearch/diamonds-seed0",
    "text_props": "redwoodresearch/text_properties",
    "gen_stories": "redwoodresearch/generated_stories",
}


class TamperingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        tokenizer,
        max_length: int = 1024,
        sensor_token=" omit",
        n_sensors: int = 3,
        train: bool = True,
        dataset_len: int | None = None,
    ):
        # TODO: allow for local loading / saving
        super().__init__()
        self.train = train
        self.name = name
        # load dataset
        hf_name = (
            TAMPERING_DATSETS[self.name]
            if self.name in TAMPERING_DATSETS
            else self.name
        )
        split = "train" if self.train else "validation"
        self.dataset = load_dataset(hf_name, split=split)
        if dataset_len:
            self.dataset = self.dataset[:dataset_len]
        # set labels
        measurements = torch.tensor(self.dataset["measurements"])
        all_measurements = torch.all(measurements, dim=1, keepdim=True)
        self.measurements = torch.concat([measurements, all_measurements], dim=1).to(
            torch.float32
        )  # (batch, nb_sensors=3 + 1)
        self.is_correct = torch.tensor(
            self.dataset["is_correct"], dtype=torch.float32
        )  # (batch,)
        self.is_clean = torch.tensor(
            self.dataset["is_clean"], dtype=torch.float32
        )  # (batch,)

        # tokenize
        self.max_length = max_length
        self.tokenized_text = tokenizer(
            self.dataset["text"],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # get sensor token ids
        self.sensor_token = sensor_token
        self.n_sensors = n_sensors
        sensor_token_id = tokenizer(self.sensor_token)["input_ids"][0]
        _batch_inds, seq_ids = torch.where(
            self.tokenized_text["input_ids"] == sensor_token_id,
        )
        self.sensor_inds = torch.reshape(seq_ids, (len(self.dataset), self.n_sensors))

    def __getitem__(self, index):
        x = {
            "input_ids": self.tokenized_text.input_ids[index],
            "attention_mask": self.tokenized_text.attention_mask[index],
            "sensor_inds": self.sensor_inds[index],
        }
        y = self.measurements[index]
        info = {"correct": self.is_correct[index], "clean": self.is_clean[index]}

        return {"x": x, "y": y, "info": info}

    def __len__(self):
        return len(self.dataset)
