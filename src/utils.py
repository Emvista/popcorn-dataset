import pickle
from torch.utils.data import DataLoader, Dataset
from torch import tensor, from_numpy, int64
from models.lightning import PL_UNIFIED
import numpy as np

from sklearn.model_selection import train_test_split

FEATURE_LEN = 768
MAX_SPAN_LEN = 17
MAX_TOKENS = 512  # 256


def get_model(config, nb_train_steps):
    return PL_UNIFIED(config, nb_train_steps=nb_train_steps)


def collate_one_for_all(batch):
    b = batch[0]
    begin_indexes = []
    end_indexes = []
    for beg, len_span in np.argwhere(b["span_labels"] > 0.5):
        begin_indexes.append(beg)
        end_indexes.append(beg + len_span)
    return {
        "ids": tensor(
            [
                [5] + x + [6] + [1] * (min(MAX_TOKENS - 2, b["max_seq_len"]) - len(x))
                for x in b["samples"]
            ]
        ),
        "mask_ids": tensor(
            [
                [1] * (len(x) + 2)
                + [0] * (min(MAX_TOKENS - 2, b["max_seq_len"]) - len(x))
                for x in b["samples"]
            ]
        ),
        "subtoken_map": tensor(b["token_word_map"], dtype=int64)
        .repeat(1536, 1)
        .transpose(1, 0),
        "span_labels": from_numpy(b["span_labels"]).double(),
        "mask": from_numpy(b["mask"]),
        "nlp_mask": from_numpy(b["nlp_mask"]),
        "nlp_labels": from_numpy(b["nlp_labels"]).double(),
        "maps": tensor(b["token_map"]),
        "len_samples": b["len_samples"],
        "begin_indexes": begin_indexes,
        "end_indexes": end_indexes,
    }


class DataSet(Dataset):
    def __init__(self, docs: list):
        self.docs = docs

    def __getitem__(self, index: int):
        return self.docs[index]

    def __len__(self):
        return len(self.docs)


def get_loaders(config: dict, nb_synthetic_set=-1):
    with open(
        config["path_train_set"],
        "rb",
    ) as tokenized_bio_file:
        train_set = pickle.load(tokenized_bio_file)

    train_set, val_set = train_test_split(
        list(train_set.values()), test_size=0.1, random_state=config["seed"]
    )
    collate_to_use = collate_one_for_all

    train_loader = DataLoader(
        DataSet(train_set),
        config["batch_size"],
        shuffle=True,
        collate_fn=collate_to_use,
    )
    val_loader = DataLoader(
        DataSet(val_set),
        config["batch_size"],
        shuffle=False,
        collate_fn=collate_to_use,
    )
    return train_loader, val_loader
