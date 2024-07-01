"""Module to format POPCORN texts for training"""

import pickle
import numpy as np

from transformers import PreTrainedTokenizerFast

from utils import ONTOLOGY

PATH_FORMATED_TRAIN_DATASET = "data/POPCORN_PAPER/train_formated.pickle"
PATH_FORMATED_TEST_DATASET = "data/POPCORN_PAPER/test_formated.pickle"
PATH_TOKENIZED_TRAIN_DATASET = "data/POPCORN_PAPER/train_tokenized.pickle"
PATH_TOKENIZED_TEST_DATASET = "data/POPCORN_PAPER/test_tokenized.pickle"

END_SENTENCE_TOKENS = [".", "▁...", "▁?", "▁!"]
CHARS_OF_INTEREST = ["-", ",", '"', ".", "(", ")"]
CHARS_OF_INTEREST_BEGIN = ["'"]

MAX_SPAN_LEN = 17
MAX_SEQ_LEN = 510
TOKENIZER = PreTrainedTokenizerFast.from_pretrained("camembert-base")
NB_CLASSES = len(ONTOLOGY)


def add_ent_types(dataset: list, sample_key: int) -> None:
    """Retrieve the mention types.

    Args:
        dataset (list): Data to process.
        text_index (int):Index of the text to process.
    """
    for entity_id, entity in enumerate(dataset[sample_key]["entities"]):
        dataset[sample_key]["entities"][entity_id]["type"] = ONTOLOGY.index(
            entity["type"]
        )


def add_text_tokens(dataset: list, text_index: str) -> None:
    """Encode text tokens with tokenizer.

    Args:
        dataset (dict)
        text_index (str): text sample key.
    """
    tokens = []
    token_word_map = []
    token_map = []
    for word, _ in dataset[text_index]["nltk_text"]:
        token_map.append(len(token_word_map))
        encoded_word = TOKENIZER.encode(word)[1:-1]
        token_word_map += [len(tokens)] * len(encoded_word)
        tokens += encoded_word
    dataset[text_index]["token_map"] = token_map
    dataset[text_index]["token_word_map"] = token_word_map
    dataset[text_index]["tokenized_text"] = tokens
    dataset[text_index]["text_len"] = len(dataset[text_index]["nltk_text"])


def get_word_offsets(nltk_text: list, label: dict) -> tuple[int, int]:
    """Add the starting and ending word indexes of the label.

    Args:
        nltk_text (list): tokenized text with offsets.
        label (dict): entity label.

    Returns:
        tuple[int, int]: start and end word indexes.
    """

    start_label_offset, end_label_offset = label["start"], label["end"]
    start_word_index = None
    for word_index, (_, (start_word_offset, end_word_offset)) in enumerate(nltk_text):
        if start_word_offset == start_label_offset:
            start_word_index = word_index
        if end_word_offset == end_label_offset:
            return start_word_index, word_index
    return None, None


def add_groundtruth(dataset: dict, text_index: str) -> None:
    """Product groundtruth matrix from annotations.

    Args:
        dataset (dict)
        text_index (str): text sample key.
    """
    groundtruth = np.zeros((dataset[text_index]["text_len"], MAX_SPAN_LEN, NB_CLASSES))
    idxs_to_delete = []
    existing_offsets = []
    for entity_id, entity in enumerate(dataset[text_index]["entities"]):
        for mention_mid, mention in enumerate(entity["mentions"]):
            start_word_index, end_word_index = get_word_offsets(
                dataset[text_index]["nltk_text"], mention
            )
            if (
                start_word_index is not None
                and end_word_index is not None
                and (start_word_index, end_word_index, entity["type"])
                not in existing_offsets
            ):
                if (
                    dataset[text_index]["nltk_text"][start_word_index][0]
                    in CHARS_OF_INTEREST + CHARS_OF_INTEREST_BEGIN
                ):
                    start_word_index += 1

                if (
                    dataset[text_index]["nltk_text"][end_word_index][0]
                    in CHARS_OF_INTEREST
                ):
                    end_word_index -= 1
                groundtruth[
                    start_word_index,
                    end_word_index - start_word_index,
                    entity["type"],
                ] = 1

                dataset[text_index]["entities"][entity_id]["mentions"][mention_mid][
                    "word_offsets"
                ] = (
                    start_word_index,
                    end_word_index - start_word_index,
                )
                existing_offsets.append(
                    (start_word_index, end_word_index, entity["type"])
                )
            else:
                idxs_to_delete = [(entity_id, mention_mid)] + idxs_to_delete
    for ent_id, mention_mid in idxs_to_delete:
        del dataset[text_index]["entities"][ent_id]["mentions"][mention_mid]
    dataset[text_index]["groundtruth"] = groundtruth


def split_tokens_into_samples(token_sentences: list) -> list:
    """Split the text tokens into a list of sequence with a maximal length.

    Args:
        token_sentences (list): _description_

    Returns:
        list: _description_
    """
    samples = []
    sample = []
    for sent in token_sentences:
        if len(sent) > MAX_SEQ_LEN:
            sample += sent
            while len(sample) > MAX_SEQ_LEN:
                samples.append(sample[:MAX_SEQ_LEN])
                sample = sample[MAX_SEQ_LEN:]
        elif len(sent) + len(sample) > MAX_SEQ_LEN:
            samples.append(sample)
            sample = [] + sent
        else:
            sample += sent
    if len(sample) > 0:
        samples.append(sample)
    return samples


def get_token_sentences(dataset: dict, text_index: str) -> list:
    """Associate the tokens to their sentence index.

    Args:
        dataset (dict)
        text_index (str): text sample key.

    Returns:
        list: _description_
    """
    token_to_sentence = []
    token_sentences = []

    token_sentence = []
    sentence_index = 0
    for token, token_id in zip(
        TOKENIZER.convert_ids_to_tokens(dataset[text_index]["tokenized_text"]),
        dataset[text_index]["tokenized_text"],
    ):
        token_to_sentence.append(sentence_index)
        token_sentence.append(token_id)
        if token in END_SENTENCE_TOKENS:
            sentence_index += 1
            token_sentences.append(token_sentence)
            token_sentence = []
    if len(token_sentence) > 0:
        token_sentences.append(token_sentence)
    dataset[text_index]["token_to_sentence"] = token_to_sentence
    return token_sentences


def add_samples(dataset: dict, text_index: str) -> None:
    """Split tokenized text into samples for training.

    Args:
        dataset (dict)
        text_index (str): text sample index
    """
    token_sentences = get_token_sentences(dataset, text_index)
    dataset[text_index]["samples"] = (
        [dataset[text_index]["tokenized_text"]]
        if len(dataset[text_index]["tokenized_text"]) <= MAX_SEQ_LEN
        else split_tokens_into_samples(token_sentences)
    )
    dataset[text_index]["max_seq_len"] = max(
        len(sample) for sample in dataset[text_index]["samples"]
    )
    dataset[text_index]["len_samples"] = [
        len(sample) for sample in dataset[text_index]["samples"]
    ]


def add_words_to_sent(dataset: dict, text_index: str) -> None:
    """Associate words to their sentence index.

    Args:
        dataset (dict)
        text_index (str): Key of the text sample.
    """
    words_to_sent = []

    current_word_index = -1
    for token_to_sent, token_to_word in zip(
        dataset[text_index]["token_to_sentence"], dataset[text_index]["token_word_map"]
    ):
        if current_word_index != token_to_word:
            words_to_sent.append(token_to_sent)
            current_word_index = token_to_word
    dataset[text_index]["words_to_sent"] = words_to_sent
    dataset[text_index]["sentences_length"] = [0]
    sent_idx = 0
    for idx in words_to_sent:
        if sent_idx == idx:
            dataset[text_index]["sentences_length"][-1] += 1
        else:
            dataset[text_index]["sentences_length"].append(1)
            sent_idx += 1
    word_to_sent_pos = []
    word_pos = 0
    current_sent = 0
    for word_to_sent_idx in dataset[text_index]["words_to_sent"]:
        if word_to_sent_idx == current_sent:
            word_to_sent_pos.append(word_pos)
            word_pos += 1
        else:
            current_sent += 1
            word_to_sent_pos.append(0)
            word_pos = 1
    dataset[text_index]["word_to_sent_pos"] = word_to_sent_pos


def add_mask(dataset: dict, text_index: str) -> None:
    """Compute Mask to prevent unpossible predictions.

    Args:
        dataset (dict)
        text_index (str): sample key in the dataset.
    """
    add_words_to_sent(dataset, text_index)
    text_data = dataset[text_index]
    begin_indices = [
        index
        for index in range(text_data["text_len"])
        for _ in range(
            min(
                text_data["sentences_length"][text_data["words_to_sent"][index]]
                - text_data["words_to_sent"][index],
                MAX_SPAN_LEN,
            )
        )
    ]

    end_indices = [
        len_span
        for index in range(text_data["text_len"])
        for len_span in range(
            min(
                text_data["sentences_length"][text_data["words_to_sent"][index]]
                - text_data["words_to_sent"][index],
                MAX_SPAN_LEN,
            )
        )
    ]
    mask = np.zeros((text_data["text_len"], MAX_SPAN_LEN))
    mask[begin_indices, end_indices] = 1
    dataset[text_index]["mask"] = mask


def prepare_data_for_model(dataset: list) -> None:
    """Add the required informations to train the models.

    Args:
        dataset (dict): dataset to process.
    """
    for text_id in dataset.keys():
        add_ent_types(dataset, text_id)
        add_text_tokens(dataset, text_id)
        add_groundtruth(dataset, text_id)
        add_samples(dataset, text_id)
        add_mask(dataset, text_id)


def main() -> None:
    with open(PATH_TOKENIZED_TRAIN_DATASET, "rb") as data_file:
        tokenized_train = pickle.load(data_file)
    prepare_data_for_model(tokenized_train)
    with open(PATH_FORMATED_TRAIN_DATASET, "wb") as data_file:
        pickle.dump(tokenized_train, data_file)

    with open(PATH_TOKENIZED_TEST_DATASET, "rb") as data_file:
        tokenized_test = pickle.load(data_file)
    prepare_data_for_model(tokenized_test)
    with open(PATH_FORMATED_TEST_DATASET, "wb") as data_file:
        pickle.dump(tokenized_test, data_file)


if __name__ == "__main__":
    main()
