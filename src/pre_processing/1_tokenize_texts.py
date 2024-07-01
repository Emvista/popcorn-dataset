import json
import pickle
import nltk

PATH_TRAIN_DATASET = "data/POPCORN_PAPER/public_train.json"
PATH_TOKENIZED_TRAIN_DATASET = "data/POPCORN_PAPER/train_tokenized.pickle"

PATH_TEST_DATASET = "data/POPCORN_PAPER/public_test.json"
PATH_TOKENIZED_TEST_DATASET = "data/POPCORN_PAPER/test_tokenized.pickle"

TOKENIZER_LANGUAGE = "french"


def rebuild_token(token: str, b_offset: int, delimiter="'") -> list:
    """Split and add samples containing unitary tokens.
    Args:
        token (str): _description_
        b_offset (int): _description_
        delimiter (str, optional): _description_. Defaults to "'".
    Returns:
        list: _description_
    """
    corrected_split = []
    subpart = ""
    for char_index, char in enumerate(token):
        if char == delimiter:
            if subpart != "":
                corrected_split += [
                    (
                        subpart,
                        (
                            (
                                b_offset
                                if len(corrected_split) == 0
                                else corrected_split[-1][1][1]
                            ),
                            b_offset + char_index,
                        ),
                    ),
                    (char, (b_offset + char_index, b_offset + char_index + 1)),
                ]
                subpart = ""
            else:
                corrected_split += [
                    (char, (b_offset + char_index, b_offset + char_index + 1)),
                ]
        else:
            subpart += char
    if len(subpart) > 0:
        corrected_split.append(
            (
                subpart,
                (
                    (
                        b_offset
                        if len(corrected_split) == 0
                        else corrected_split[-1][1][1]
                    ),
                    b_offset + char_index + 1,
                ),
            )
        )
    return corrected_split


def word_tokenize(text: str) -> list:
    """Use NLTK Tokenizer to get unit tokens.

    Args:
        text (str): text

    Returns:
        list: list of tokens
    """
    return [
        token.replace("``", '"').replace("''", '"')
        for token in nltk.word_tokenize(text, language=TOKENIZER_LANGUAGE)
    ]


def spans(txt: str) -> (str, (int, int)):
    """Split the text spans.

    Args:
        txt (str): _description_

    Yields:
        (Tuple[str, Tuple[int, int]]): _description_
    """
    tokens = word_tokenize(txt)
    offset = 0
    for token in tokens:
        new_offset = txt.find(token, offset)
        if new_offset - offset > len(token) and token == '"':
            token = "''"
            new_offset = txt.find(token, offset)
        elif new_offset == -1:
            token = "''"
            new_offset = txt.find(token, offset)
        yield token, (new_offset, new_offset + len(token))
        offset = new_offset + len(token)


def correct_splitting(text: list, delimiter="'") -> str:
    """Split correctly each tokens given a delimiter of interest.

    Args:
        text (list): _description_
        delimiter (str, optional): _description_. Defaults to "'".

    Returns:
        str: tokens with their offsets.
    """
    corrected_text = []
    for token, (b_offset, e_offset) in text:
        if token == "":
            continue
        if delimiter in token and len(token) > 1:
            corrected_text += rebuild_token(token, b_offset, delimiter)
        else:
            corrected_text.append((token, (b_offset, e_offset)))
    return corrected_text


def get_nltk_tokenization(text: str) -> list:
    """Split the text into words with their offsets.

    Args:
        text (str): text content.

    Returns:
        list: Ordered list of text words with their offsets.
    """
    cleaned_text = correct_splitting(list(spans(text)), "'")
    cleaned_text = correct_splitting(cleaned_text, "-")
    return cleaned_text


def tokenize_set(dataset: list) -> list:
    """Tokenize dataset texts.

    Args:
        dataset (list): data samples.
    Returns:
        list: dataset with tokenized texts.
    """
    for text_id in dataset.keys():
        dataset[text_id]["nltk_text"] = get_nltk_tokenization(dataset[text_id]["text"])
    return dataset


def main():
    with open(PATH_TRAIN_DATASET, "r", encoding="utf-8") as train_file:
        train_set = json.load(train_file)
    train_set = tokenize_set(train_set)
    with open(PATH_TOKENIZED_TRAIN_DATASET, "wb") as train_file:
        pickle.dump(train_set, train_file)

    with open(PATH_TEST_DATASET, "r", encoding="utf-8") as test_file:
        test_set = json.load(test_file)
    test_set = tokenize_set(test_set)
    with open(PATH_TOKENIZED_TEST_DATASET, "wb") as test_file:
        pickle.dump(test_set, test_file)


if __name__ == "__main__":
    main()
