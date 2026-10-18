"""
Lab 2.

BPE and machine translation evaluation
"""

# pylint:disable=unused-argument
from typing import Sequence
import json


def prepare_word(
    raw_word: str, start_of_word: str | None, end_of_word: str | None
) -> tuple[str, ...] | None:
    """
    Tokenize a word into characters and attach optional boundary tokens.

    Args:
        raw_word (str): Original word
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str | None): A token that signifies the end of word

    Returns:
        tuple[str, ...] | None: Preprocessed word

    In case of corrupt input arguments, None is returned
    """
    if not all((
        isinstance(raw_word, str),
        isinstance(start_of_word, (str, type(None))),
        isinstance(end_of_word, (str, type(None))),
    )):
        return None
    prepared_word = [letter for letter in raw_word]
    if start_of_word is not None:
        prepared_word.insert(0, start_of_word)
    if end_of_word is not None:
        prepared_word.append(end_of_word)
    return tuple(prepared_word)



def collect_frequencies(
    text: str, start_of_word: str | None, end_of_word: str
) -> dict[tuple[str, ...], int] | None:
    """
    Count number of occurrences of each word.

    Args:
        text (str): Original text with no preprocessing
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str): A token that signifies the end of word

    Returns:
        dict[tuple[str, ...], int] | None: Dictionary where key - preprocessed word,
            value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if not all((
        isinstance(text, str),
        isinstance(start_of_word, (str, type(None))),
        isinstance(end_of_word, str),
    )):
        return None
    freqs = {}
    for word in text.split():
        actual = prepare_word(word, start_of_word, end_of_word)
        if actual is None:
            return None
        freqs[actual] = freqs.get(actual, 0) + 1
    return freqs



def count_tokens_pairs(
    word_frequencies: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int] | None:
    """
    Count number of occurrences of each pair of subsequent tokens.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences

    Returns:
        dict[tuple[str, str], int] | None: A dictionary where
            key - token pair, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(word_frequencies, dict):
        return None
    for key in word_frequencies.keys():
        if not isinstance(key, tuple):
            return None
        for v in key:
            if not isinstance(v, str):
                return None
    for value in word_frequencies.values():
        if not isinstance(value, int):
            return None
    new_dict = {}
    for tokens, count in word_frequencies.items():
        for x in range(len(tokens) - 1):
            pair = (tokens[x], tokens[x + 1])
            new_dict[pair] = new_dict.get(pair, 0) + count
    return new_dict



def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Update word frequency dictionary by replacing a pair of tokens with a merged one.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        pair (tuple[str, str]): A pair of tokens to be merged

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if not all((
        isinstance(word_frequencies, dict),
        isinstance(pair, tuple),
    )):
        return None
    for key, value in word_frequencies.items():
        if not all((
            isinstance(key, tuple),
            isinstance(value, int),
        )):
            return None
        for x in key:
            if not isinstance(x, str):
                return None
    for y in pair:
        if not isinstance(y, str):
            return None
    another_dict = {}
    for token, count in word_frequencies.items():
        new_tokens = []
        i = 0
        while i < len(token):
            if (i + 1 < len(token) and
                token[i] == pair[0] and
                token[i + 1] == pair[1]):
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(token[i])
                i += 1
        new_keys = tuple(new_tokens)
        if new_keys in another_dict:
            another_dict[new_keys] += count
        else:
            another_dict[new_keys] = count
    return another_dict



def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Create required number of new tokens by merging existing ones.

    Args:
        word_frequencies (dict[tuple[str, ...], int] | None): A dictionary where
            key - preprocessed word, value - number of occurrences
        num_merges (int): Required number of new tokens

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if not all((
        isinstance(word_frequencies, dict),
        isinstance(num_merges, int),
    )):
        return None
    for k, v in word_frequencies.items():
        if not all((
            isinstance(k, tuple),
            isinstance(v, int),
        )):
            return None
        for s in k:
            if not isinstance(s, str):
                return None
    d = word_frequencies.copy()
    for _ in range(num_merges):
        counted_tokens = count_tokens_pairs(d)
        if counted_tokens is None or not counted_tokens:
            break
        sorted_tokens = sorted(
            counted_tokens.items(),
            key = lambda x: (
                -x[1],
                -len(x[0][0] + x[0][1]),
                x[0][0] + x[0][1]
            )
        )
        chosen_one = sorted_tokens[0][0]
        d = merge_tokens(d, chosen_one)
        if d is None:
            return None
    return d



def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establish correspondence between tokens and its integer identifier.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        unknown_token (str): A token to signify an unknown token

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if not all((
            isinstance(word_frequencies, (dict)),
            isinstance(unknown_token, str),
        )):
            return None
    for k, v in word_frequencies.items():
        if not all((
            isinstance(k, tuple),
            isinstance(v, int),
        )):
            return None
        for s in k:
            if not isinstance(s, str):
                return None
    tokens = set()
    for word in word_frequencies:
        tokens.update(word)
    words = {t for t in tokens if len(t) > 1}
    letters = {t for t in tokens if len(t) == 1}
    for word in words:
        for letter in word:
            letters.add(letter)
    d = {}
    sorted_tokens = sorted(tokens, key = lambda x: (-len(x), x))
    sorted_letters = sorted(letters)
    for i, t in enumerate(sorted_tokens):
        d[t] = i
    d[unknown_token] = len(sorted_tokens)
    for i, l in enumerate(sorted_letters, start = len(sorted_tokens) + 1):
        d[l] = i
    return d



def decode(
    encoded_text: Sequence[int] | None,
    vocabulary: dict[str, int] | None,
    end_of_word_token: str | None,
) -> str | None:
    """
    Translate encoded sequence into decoded one.

    Args:
        encoded_text (Sequence[int] | None): A sequence of token identifiers
        vocabulary (dict[str, int] | None): A dictionary where
            key - token, value - identifier
        end_of_word_token (str | None): An end-of-word token

    Returns:
        str | None: Decoded sequence

    In case of corrupt input arguments, None is returned
    """
    if not all((
        isinstance(encoded_text, (tuple, list)),
        isinstance(vocabulary, dict),
        isinstance(end_of_word_token, (str, type(None))),
    )):
        return None
    if len(encoded_text) == 0:
        return None
    for k, v in vocabulary.items():
        if not all((
            isinstance(k, str),
            isinstance(v, int),
            not isinstance(v, bool),
        )):
            return None
    for char in encoded_text:
        if not isinstance(char, int) or isinstance(char, bool):
            return None
    reverse = {i: token for token, i in vocabulary.items()}
    if len(set(vocabulary.values())) != len(vocabulary):
        return None
    text = []
    for x in encoded_text:
        if x not in reverse:
            return None
        text.append(reverse[x])
    for j in range(len(text)):
        if end_of_word_token is not None and end_of_word_token in text[j]:
            text[j] = text[j].replace(end_of_word_token, ' ')
    actual_text = ''.join(text)
    return actual_text



def tokenize_word(
    word: tuple[str, ...], vocabulary: dict[str, int], end_of_word: str | None, unknown_token: str
) -> list[int] | None:
    """
    Split word into tokens.

    Args:
        word (tuple[str, ...]): Preprocessed word
        vocabulary (dict[str, int]): A dictionary where key - token, value - identifier
        end_of_word (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments, None is returned
    """
    if not all((
        isinstance(word, tuple),
        isinstance(vocabulary, dict),
        isinstance(end_of_word, (str, type(None))),
        isinstance(unknown_token, str),
    )):
        return None
    for s in word:
        if not isinstance(s, str):
            return None
    for k, v in vocabulary.items():
        if not all((isinstance(k, str),
                   isinstance(v, int)
        )):
            return None
    if unknown_token not in vocabulary:
            return None
    text = ''.join(word)
    if end_of_word is not None and not text.endswith(end_of_word):
        text += end_of_word
    tokens = [t for t in vocabulary if t != unknown_token]
    sorted_tokens = sorted(tokens, key = lambda t: (-len(t), t))
    segments = [text]
    for t in sorted_tokens:
        new_segments = []
        for s in segments:
            if s in vocabulary:
                new_segments.append(s)
                continue
            parts = s.split(t)
            for idx, part in enumerate(parts):
                if part:
                    new_segments.append(part)
                if idx < len(parts) - 1:
                    new_segments.append(t)
        segments = new_segments
    result = []
    for s in segments:
        if s in vocabulary and s != unknown_token:
            result.append(vocabulary[s])
        else:
            result.extend([vocabulary[unknown_token]] * len(s))
    return result



def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Read and retrieve dictionary of type <token: identifier>.

    Args:
        vocab_path (str): A path to the saved vocabulary

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(vocab_path, str):
        return None
    with open(vocab_path, 'r', encoding='utf-8') as file:
        d = json.load(file)
    if not isinstance(d, dict):
        return None
    if not all(
        isinstance(k, str) and
        isinstance(v, int) for k, v in d.items()
    ):
        return None
    return d



def encode(
    original_text: str,
    vocabulary: dict[str, int] | None,
    start_of_word_token: str | None,
    end_of_word_token: str | None,
    unknown_token: str,
) -> list[int] | None:
    """
    Translate original text into a sequence of token identifiers.

    Args:
        original_text (str): Original text
        vocabulary (dict[str, int] | None): A dictionary where key - token, value - identifier
        start_of_word_token (str | None): A start-of-word token
        end_of_word_token (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if not all((
        isinstance(original_text, str),
        isinstance(vocabulary, dict),
        isinstance(start_of_word_token, (str, type(None))),
        isinstance(end_of_word_token, (str, type(None))),
        isinstance(unknown_token, str),
    )):
        return None
    for k, v in vocabulary.items():
        if not all((isinstance(k, str), isinstance(v, int))):
            return None
    words = original_text.split()
    if not words:
        return None
    result = []
    for raw_word in words:
        prepared_word = prepare_word(
            raw_word,
            start_of_word_token,
            end_of_word_token
        )
        if prepared_word is None:
            return None
        tokenized = tokenize_word(
            prepared_word,
            vocabulary,
            end_of_word_token,
            unknown_token
        )
        if tokenized is None:
            return None
        result.extend(tokenized)
    return result


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extract n-grams from the given sequence.

    Args:
        text (str): Original text
        order (int): Required number of elements in a single n-gram

    Returns:
        list[tuple[str, ...]] | None: A sequence of n-grams

    In case of corrupt input arguments, None is returned
    """


def calculate_precision(
    actual: Sequence[tuple[str, ...]], reference: Sequence[tuple[str, ...]]
) -> float | None:
    """
    Compare two sequences by virtue of Precision metric.

    Args:
        actual (Sequence[tuple[str, ...]]): Predicted sequence of n-grams
        reference (Sequence[tuple[str, ...]]): Expected sequence of n-grams

    Returns:
        float | None: Value of Precision metric

    In case of corrupt input arguments, None is returned.
    """


def calculate_geo_mean(precisions: Sequence[float], max_order: int) -> float | None:
    """
    Compute geometric mean of sequence of values.

    Args:
        precisions (Sequence[float]): A sequence of Precision values
        max_order (int): Maximum length of n-gram considered

    Returns:
        float | None: A value of geometric mean of Precision metric

    In case of corrupt input arguments, None is returned
    """


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compare two sequences by virtue of BLEU metric.

    Args:
        actual (str | None): Predicted sequence
        reference (str): Expected sequence
        max_order (int, optional): Max length of n-gram to consider for comparison. Defaults to 3

    Returns:
        float | None: A value of BLEU metric

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
