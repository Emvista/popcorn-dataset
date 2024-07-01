import pickle
import numpy as np
from tqdm import tqdm
from utils import ONTOLOGY, ONTOLOGY_RELATIONS


PATH_FORMATED_TRAIN_DATASET = "data/POPCORN_PAPER/train_formated.pickle"
PATH_FORMATED_TEST_DATASET = "data/POPCORN_PAPER/test_formated.pickle"
PATH_MASKED_TRAIN_DATASET = "data/POPCORN_PAPER/train_nlp.pickle"
PATH_MASKED_TEST_DATASET = "data/POPCORN_PAPER/test_nlp.pickle"


MAX_SPAN_LEN = 17
MAX_SEQ_LEN = 510


def get_span_labels(groundtruth: np.array) -> np.array:
    cleaned_groundtruth = np.zeros(
        (groundtruth.shape[0], groundtruth.shape[1]),
        dtype=bool,
    )
    for i, j, k in np.argwhere(groundtruth > 0):
        cleaned_groundtruth[i, j] = max(cleaned_groundtruth[i, j], groundtruth[i, j, k])
    return cleaned_groundtruth


def add_types(
    nlp_labels: np.array, groundtruth: np.array, offset_mentions: list
) -> np.array:
    """
    Add entities types to the nlp groundtruth matrix.
    Args:
        nlp_labels (np.array)
        groundtruth (np.array): groundtruth ner matrix
        offset_mentions (list)

    Returns:
        np.array: 3D matrix with one value for types possessed by an entity.
    """
    for index_ment, (begin_ment, end_ment) in enumerate(offset_mentions):
        for sup in np.argwhere(groundtruth[begin_ment, end_ment] > 0.5):
            nlp_labels[index_ment, index_ment, sup[0]] = 1
    return nlp_labels


def add_relations(nlp_labels: np.array, text_data: dict) -> np.array:
    """Add relations to the nlp matrix.

    Args:
        nlp_labels (np.array): _description_
        text_data (dict): _description_

    Returns:
        np.array: _description_
    """
    for start_cluster, predicate, end_cluster in text_data["relations"]:
        for start_mention in text_data["entities"][start_cluster]["mentions"]:
            for end_mention in text_data["entities"][end_cluster]["mentions"]:
                nlp_labels[
                    start_mention["mention_idx"],
                    end_mention["mention_idx"],
                    len(ONTOLOGY) + ONTOLOGY_RELATIONS.index(predicate),
                ] = 1
    return nlp_labels


def add_coreferences(nlp_labels: np.array, text_data: dict) -> np.array:
    """Add coreference label to nlp matrix.

    Args:
        nlp_labels (np.array): _description_
        text_data (dict): _description_

    Returns:
        np.array: _description_
    """
    for ent in text_data["entities"]:
        for first_mention in ent["mentions"]:
            for second_mention in ent["mentions"]:
                nlp_labels[
                    first_mention["mention_idx"], second_mention["mention_idx"], -1
                ] = 1
    return nlp_labels


def get_nlp_mask(nlp_shapes: tuple, nb_types: int) -> np.array:
    """_summary_

    Args:
        nlp_shapes (tuple): _description_
        nb_types (int): _description_

    Returns:
        np.array: _description_
    """
    nlp_mask = np.ones(nlp_shapes, dtype=bool)
    for i in range(nlp_shapes[0]):
        nlp_mask[i, :, :nb_types] = 0
        # nlp_mask[i, i, nb_types:-3] = 0  # Gender male, gender female and co-references can be predicted.
        nlp_mask[i, i, :nb_types] = 1
    return nlp_mask


def add_mention_index(offset_mentions: list, text_idx: str, dataset: dict) -> None:
    """Add their index to entity mentions.

    Args:
        offset_mentions (list): mentions offsets.
        text_idx (str): text sample key.
        dataset (dict)
    """
    for mtion_idx, (start_idx, end_idx) in enumerate(offset_mentions):
        for ent_idx, ent in enumerate(dataset[text_idx]["entities"]):
            for mention_idx, mention in enumerate(ent["mentions"]):
                if "word_offsets" in mention:
                    start_offset, end_offset = mention["word_offsets"]
                    if start_offset == start_idx and end_offset == end_idx:
                        dataset[text_idx]["entities"][ent_idx]["mentions"][mention_idx][
                            "mention_idx"
                        ] = mtion_idx


def add_data_for_model(dataset: dict) -> None:
    """Add the required informations to train the models for nlp tasks.

    Args:
        dataset (dict): dataset to process.
    """
    for text_index, text_data in tqdm(dataset.items()):
        offset_mentions = sorted(
            list(set((i, j) for i, j, k in np.argwhere(text_data["groundtruth"] > 0.5)))
        )
        nlp_labels = np.zeros(
            (
                len(offset_mentions),
                len(offset_mentions),
                len(ONTOLOGY) + len(ONTOLOGY_RELATIONS) + 1,  # +1 Is for coreference.
            ),
            dtype=bool,
        )
        add_mention_index(offset_mentions, text_index, dataset)
        nlp_labels = add_types(nlp_labels, text_data["groundtruth"], offset_mentions)
        nlp_labels = add_coreferences(nlp_labels, text_data)
        nlp_labels = add_relations(nlp_labels, text_data)
        dataset[text_index]["span_labels"] = get_span_labels(text_data["groundtruth"])
        dataset[text_index]["nlp_mask"] = get_nlp_mask(
            nlp_labels.shape, text_data["groundtruth"].shape[2]
        )
        dataset[text_index]["nlp_labels"] = nlp_labels


def main() -> None:
    with open(PATH_FORMATED_TRAIN_DATASET, "rb") as data_file:
        train_set = pickle.load(data_file)
    add_data_for_model(train_set)
    with open(PATH_MASKED_TRAIN_DATASET, "wb") as data_file:
        pickle.dump(train_set, data_file)

    with open(PATH_FORMATED_TEST_DATASET, "rb") as data_file:
        test_set = pickle.load(data_file)
    add_data_for_model(test_set)
    with open(PATH_MASKED_TEST_DATASET, "wb") as data_file:
        pickle.dump(test_set, data_file)


if __name__ == "__main__":
    main()
