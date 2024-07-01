import json
import os
import argparse
from models.lightning import PL_UNIFIED
import torch
import pickle
import numpy as np

from torch import from_numpy, tensor
from collections import defaultdict
from tqdm import tqdm

from pre_processing.utils import (
    ONTOLOGY,
    ONTOLOGY_RELATIONS,
    TYPE_TO_PARENT,
    VALID_COMBINATIONS,
)

PATH_CONFIG = "data/POPCORN_PAPER/models/config/teacher/seed_"
MAX_SPAN_LEN = 17
MAX_SEQ_LEN = 510
NB_SEED = 5


def load_model():
    device = torch.device("cuda:0")
    model = PL_UNIFIED.load_from_checkpoint(
        checkpoint_path=f"{config['path_dir_checkpoint']}/{config['model_name']}.ckpt",
        config=config,
        device=device,
    )

    model.to(device)
    model.eval()
    return model, device


def preprocess_text(text_data: dict):
    begin_indexes = []
    end_indexes = []
    for beg, len_span in np.argwhere(text_data["span_labels"] > 0.5):
        begin_indexes.append(beg)
        end_indexes.append(beg + len_span)
    return {
        "ids": tensor(
            [
                [5]
                + x
                + [6]
                + [1] * (min(MAX_SEQ_LEN, text_data["max_seq_len"]) - len(x))
                for x in text_data["samples"]
            ]
        ),
        "mask_ids": tensor(
            [
                [1] * (len(x) + 2)
                + [0] * (min(MAX_SEQ_LEN, text_data["max_seq_len"]) - len(x))
                for x in text_data["samples"]
            ]
        ),
        "subtoken_map": tensor(text_data["token_word_map"], dtype=torch.int64)
        .repeat(1536, 1)
        .transpose(1, 0),
        "mask": from_numpy(text_data["mask"]),
        "nlp_mask": from_numpy(text_data["nlp_mask"]),
        "nlp_labels": from_numpy(text_data["nlp_labels"]).double(),
        "maps": tensor(text_data["token_map"]),
        "len_samples": text_data["len_samples"],
        "begin_indexes": begin_indexes,
        "end_indexes": end_indexes,
    }


def format_triple(subjects, predicate, objects, entities):
    for subj_idx, subj_key in subjects:
        for obj_idx, obj_key in objects:
            if (
                TYPE_TO_PARENT[ONTOLOGY[entities[subj_idx]["type"]]],
                predicate,
                TYPE_TO_PARENT[ONTOLOGY[entities[obj_idx]["type"]]],
            ) in VALID_COMBINATIONS:
                return (subj_key, predicate, obj_key)

    return None


def format_relations(pred_rels: torch.tensor, text_data: dict) -> list:
    relations = set()
    mention_idx_to_ent_key = defaultdict(list)
    for ent_idx, ent in enumerate(text_data["entities"]):
        for mention in ent["mentions"]:
            mention_idx_to_ent_key[mention["mention_idx"]].append((ent_idx, ent["id"]))
    positive_rels = torch.argwhere(pred_rels[:, :, len(ONTOLOGY) : -1] > 0)
    for rel in positive_rels:
        subject_ent_indexes = mention_idx_to_ent_key[rel[0].item()]
        object_ent_indexes = mention_idx_to_ent_key[rel[1].item()]
        predicate = ONTOLOGY_RELATIONS[rel[2]]
        triple = format_triple(
            subject_ent_indexes, predicate, object_ent_indexes, text_data["entities"]
        )
        if triple is not None and (
            (triple[1] in ["GENDER_MALE", "GENDER_FEMALE"] and triple[0] == triple[2])
            or (
                triple[1] not in ["GENDER_MALE", "GENDER_FEMALE"]
                and triple[0] != triple[2]
            )
        ):
            relations.add(triple)
    return list(relations)


def predict_relations(test_set: dict, model: PL_UNIFIED, device) -> None:
    with open(config["path_test_json"], "r") as test_file:
        test_output = json.load(test_file)
    for text_key, text_data in tqdm(test_set.items()):
        with torch.no_grad():
            pred_rels = model.predict_relations(preprocess_text(text_data))
        test_output[text_key]["relations"] = format_relations(pred_rels, text_data)
    if not os.path.exists(config["path_output"]):
        os.mkdir(config["path_output"])
    with open(
        config["path_output"] + "relations_" + config["model_name"] + ".json",
        "w",
    ) as predicted_file:
        json.dump(test_output, predicted_file)


def format_mentions(
    begin_indexes: list,
    end_indexes: list,
    pred_interactions: torch.tensor,
    text_data: dict,
) -> list:
    entities = []
    for index_first_span, index_second_span, pred_type in torch.argwhere(
        pred_interactions[:, :, : len(ONTOLOGY)] > 0
    ):
        if index_first_span == index_second_span:
            start = text_data["nltk_text"][begin_indexes[index_first_span]][1][0]
            end = text_data["nltk_text"][end_indexes[index_first_span]][1][1]
            entities.append(
                {
                    "type": ONTOLOGY[pred_type.item()],
                    "mentions": [
                        {
                            "value": text_data["text"][start:end],
                            "start": start,
                            "end": end,
                        }
                    ],
                }
            )
    return entities


def predict_all(test_set: dict, model: PL_UNIFIED, device) -> None:
    with open(config["path_test_json"], "r") as test_file:
        test_output = json.load(test_file)
    for text_key, text_data in tqdm(test_set.items()):
        with torch.no_grad():
            begin_indexes, end_indexes, pred_interactions = model.predict_all(
                preprocess_text(text_data)
            )
        test_output[text_key]["entities"] = format_mentions(
            begin_indexes, end_indexes, pred_interactions, text_data
        )
    if not os.path.exists(config["path_output"]):
        os.mkdir(config["path_output"])
    with open(
        config["path_output"] + config["model_name"] + ".json",
        "w",
    ) as predicted_file:
        json.dump(test_output, predicted_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=PATH_CONFIG,
        help="Path to the config file for training",
    )
    args = parser.parse_args()
    for seed_idx in range(1, NB_SEED + 1):
        path_config = args.config_path + str(seed_idx) + ".json"
        with open(path_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(
            config["path_test_set"],
            "rb",
        ) as test_file:
            test_set = pickle.load(test_file)
        model, device = load_model()
        predict_relations(test_set, model, device)
        # predict_coreference(test_set) TODO
        predict_all(test_set, model, device)
