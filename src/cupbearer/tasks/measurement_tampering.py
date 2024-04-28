import numpy as np
from torch.utils.data import Dataset, Subset

from cupbearer.data import TamperingDataset
from cupbearer.models import TamperingPredictionTransformer

from .task import Task


def trusted_data_mask(data: Dataset):
    clean_mask = np.array(
        [info["clean"] for x, y, *info in data]
    )
    return clean_mask


def anomalous_data_mask(data: Dataset):
    tampered_mask = np.array(
        [
            y[-1] != info["correct"]
            for x, y, *info in data
        ]
    )
    return tampered_mask
    # check if all(measurments) = is correct


def measurement_tampering(
    model: TamperingPredictionTransformer,
    train_data: Dataset,
    test_data: Dataset,
    clean_test_weight: float = 0.5,
):
    train_trusted_data_mask = trusted_data_mask(train_data)
    trusted_data = Subset(train_data, np.where(train_trusted_data_mask)[0])
    untrusted_train_data = Subset(train_data, np.where(~train_trusted_data_mask)[0])

    # TODO: add in/out easy/hard trusted/untrusted to task and eval
    # test_trusted_data_mask = trusted_data_mask(test_data)

    test_anomalous_data_mask = anomalous_data_mask(test_data)
    clean_test_data = Subset(test_data, np.where(~test_anomalous_data_mask)[0])
    anomalous_test_data = Subset(test_data, np.where(test_anomalous_data_mask)[0])

    return Task.from_separate_data(
        model=model,
        trusted_data=trusted_data,
        clean_test_data=clean_test_data,
        anomalous_test_data=anomalous_test_data,
        untrusted_train_data=untrusted_train_data,
        clean_test_weight=clean_test_weight,
    )
