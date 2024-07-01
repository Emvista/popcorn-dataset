"""Script to evaluate mention predictions."""

import os
import argparse
import json
from collections import defaultdict
from utils import ONTOLOGY_EVENTS


PATH_CONFIG = "data/POPCORN_PAPER/models/config/teacher/seed_"
PATH_TEST_FILE = "./data/POPCORN_PAPER/public_test.json"  # TODO Modify with actual path
PATH_PREDICTIONS = "./data/POPCORN_PAPER/output/"
NB_SEED = 5


def evaluate(test: dict, predictions: dict) -> tuple[float, float]:
    """Measure mention Extraction Macro and Micro F1.

    Args:
        test (dict): groundtruth annotations
        predictions (dict): predicted annotations
    """
    false_negatives, false_positives, true_positives = [], [], []
    formated_errors = {}
    for text_id, text_data in test.items():
        test_mentions = set(
            [
                (ent["type"], mention["start"], mention["end"])
                for ent in text_data["entities"]
                for mention in ent["mentions"]
                if ent["type"] in ONTOLOGY_EVENTS
            ]
        )  # tuples are list in json hence unhashable
        predicted_mentions = set(
            [
                (ent["type"], mention["start"], mention["end"])
                for ent in predictions[text_id]["entities"]
                for mention in ent["mentions"]
                if ent["type"] in ONTOLOGY_EVENTS
            ]
        )
        false_positives += list(predicted_mentions.difference(test_mentions))
        false_negatives += list(test_mentions.difference(predicted_mentions))
        true_positives += list(test_mentions.intersection(predicted_mentions))

    # Dispatch FP, FN, TP per mention type
    fn_mentions = defaultdict(list)
    tp_mentions = defaultdict(list)
    fp_mentions = defaultdict(list)
    for mention in false_negatives:
        fn_mentions[mention[0]].append(mention)
    for mention in false_positives:
        fp_mentions[mention[0]].append(mention)
    for mention in true_positives:
        tp_mentions[mention[0]].append(mention)
    # Compute F1 for each mention type
    f1s = dict()
    for predicate in set(tp_mentions.keys()).union(fn_mentions.keys()):
        precision = (
            0
            if len(fp_mentions[predicate] + tp_mentions[predicate]) == 0
            else len(tp_mentions[predicate])
            / len(tp_mentions[predicate] + fp_mentions[predicate])
        )
        recall = (
            0
            if len(fn_mentions[predicate] + tp_mentions[predicate]) == 0
            else len(tp_mentions[predicate])
            / len(tp_mentions[predicate] + fn_mentions[predicate])
        )
        f1s[predicate] = (
            0
            if recall + precision == 0
            else 2 * precision * recall / (precision + recall)
        )
    macro_f1 = sum(f1s.values()) / len(f1s)
    nb_tp = sum(len(tps) for tps in tp_mentions.values())
    nb_fp = sum(len(fps) for fps in fp_mentions.values())
    nb_fn = sum(len(fns) for fns in fn_mentions.values())
    precision = 0 if nb_tp + nb_fp == 0 else nb_tp / (nb_tp + nb_fp)
    recall = 0 if nb_tp + nb_fn == 0 else nb_tp / (nb_tp + nb_fn)
    micro_f1 = (
        0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    return micro_f1, macro_f1


def main(path_test_file: str, path_predictions: str) -> None:
    """Load and evaluate predicted_mentions

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
