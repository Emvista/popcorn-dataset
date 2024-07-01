"""Script to evaluate relation extraction."""

import os
import argparse
import json
from collections import defaultdict
from pytorch_lightning import seed_everything

PATH_CONFIG = "data/POPCORN_PAPER/models/config/teacher/seed_"
PATH_TEST_FILE = "./data/POPCORN_PAPER/public_test.json"  # TODO Modify with actual path
PATH_PREDICTIONS = "./data/POPCORN_PAPER/output/relations_"
NB_SEED = 5


def evaluate(test: dict, predictions: dict) -> tuple[float, float]:
    """Measure Relation Extraction Macro and Micro F1.

    Args:
        test (dict): groundtruth annotations
        predictions (dict): predicted annotations
    """
    false_negatives, false_positives, true_positives = [], [], []
    for text_id, text_data in test.items():
        test_rels = set(
            [tuple(rel) for rel in text_data["relations"]]
        )  # tuples are list in json hence unhashable
        predicted_relations = set(
            [tuple(rel) for rel in predictions[text_id]["relations"]]
        )  # tuples are list in json hence unhashable
        false_positives += list(predicted_relations.difference(test_rels))
        false_negatives += list(test_rels.difference(predicted_relations))
        true_positives += list(test_rels.intersection(predicted_relations))

    # Dispatch FP, FN, TP per relation type
    fn_rels = defaultdict(list)
    tp_rels = defaultdict(list)
    fp_rels = defaultdict(list)
    for relation in false_negatives:
        fn_rels[relation[1]].append(relation)
    for relation in false_positives:
        fp_rels[relation[1]].append(relation)
    for relation in true_positives:
        tp_rels[relation[1]].append(relation)

    f1s = dict()
    for predicate in set(tp_rels.keys()).union(fn_rels.keys()):
        precision = (
            0
            if len(fp_rels[predicate] + tp_rels[predicate]) == 0
            else len(tp_rels[predicate]) / len(tp_rels[predicate] + fp_rels[predicate])
        )
        recall = (
            0
            if len(fn_rels[predicate] + tp_rels[predicate]) == 0
            else len(tp_rels[predicate]) / len(tp_rels[predicate] + fn_rels[predicate])
        )
        f1s[predicate] = (
            0
            if recall + precision == 0
            else 2 * precision * recall / (precision + recall)
        )
    macro_f1 = sum(f1s.values()) / len(f1s)
    nb_tp = sum(len(tps) for tps in tp_rels.values())
    nb_fp = sum(len(fps) for fps in fp_rels.values())
    nb_fn = sum(len(fns) for fns in fn_rels.values())
    precision = 0 if nb_tp + nb_fp == 0 else nb_tp / (nb_tp + nb_fp)
    recall = 0 if nb_tp + nb_fn == 0 else nb_tp / (nb_tp + nb_fn)
    micro_f1 = (
        0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    return micro_f1, macro_f1


def main(path_test_file: str, path_predictions: str) -> tuple[float, float]:
    """Load and evaluate predicted_relations

    Args:
        path_test_file (str): Path of the test file
        path_predictions (str): Path of the prediction file
    """
    with open(path_test_file, "r", encoding="utf-8") as test_file:
        test = json.load(test_file)
    with open(path_predictions, "r", encoding="utf-8") as test_file:
        predictions = json.load(test_file)
    micro_f1, macro_f1 = evaluate(test, predictions)
    return micro_f1, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_test_file",
        type=str,
        default=PATH_TEST_FILE,
        help="Path to the test data",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=PATH_CONFIG,
        help="Path to the config file for training",
    )
    parser.add_argument(
        "--path_prediction_file",
        type=str,
        default=PATH_PREDICTIONS,
        help="Path to the prediction file",
    )
    args = parser.parse_args()

    micro_f1s, macro_f1s = [], []
    for seed_idx in range(1, NB_SEED + 1):
        path_config = args.config_path + str(seed_idx) + ".json"
        with open(path_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        seed_everything(config["seed"])
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
        if os.path.exists(args.path_prediction_file + config["model_name"] + ".json"):
            micro_f1, macro_f1 = main(
                args.path_test_file,
                args.path_prediction_file + config["model_name"] + ".json",
            )
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
    print(
        f"Micro-F1 : {sum(micro_f1s)/len(micro_f1s)}, Macro-F1 : {sum(macro_f1s)/len(macro_f1s)}"
    )
