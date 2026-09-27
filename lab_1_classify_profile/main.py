"""
Lab 1.

Language detection
"""

# pylint:disable=unused-argument
from typing import Sequence

FreqDictType = dict[str, float]
"Frequency dictionary. Contains pairs of token and its frequency."
ProfileType = tuple[str, FreqDictType, int]
"Language profile of a text. Contains language name, frequency dictionary and number of tokens."
# Mark 4.


def tokenize(text: str) -> Sequence[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation and other symbols from words

    Args:
       text (str): Text

    Returns:
        Sequence[str] | None: Sequence of lower-cased tokens without punctuation.
        Returns None if input text is not a string.
    """
    if not isinstance(text, str):
        return None
    cleaned_parts = []
    for item in text.lower():
        if item.isalpha() or item.isspace():
            cleaned_parts.append(item)
    cleaned_text = ''.join(cleaned_parts)
    tokens = cleaned_text.split()
    return tokens



def remove_stop_words(tokens: Sequence[str], stop_words: Sequence[str]) -> Sequence[str] | None:
    """
    Removes stop words

    Args:
        tokens (Sequence[str]): Sequence of tokens
        stop_words (Sequence[str]): Sequence of stop words (can be empty)
    Returns:
        Sequence[str] | None: Sequence of tokens without stop words.
        Returns None in case of incorrect input types.
    """
    if not isinstance(tokens, (list, tuple)) or not isinstance(stop_words, (list, tuple)):
        return None
    for word in tokens:
        if not isinstance(word,str):
            return None
    for word in stop_words:
        if not isinstance(word, str):
            return None
    actual_words = []
    for word in tokens:
        if word not in stop_words:
            actual_words.append(word)
    return actual_words


def calculate_frequencies(tokens: Sequence[str]) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens

    Args:
        tokens (Sequence[str]): Sequence of tokens
    Returns:
        dict[str, float] | None: Dictionary with frequencies.
        Returns None in case of incorrect input types.
    """
    if not isinstance(tokens, (list, tuple)):
        return None
    for word in tokens:
        if not isinstance(word, str):
            return None
    freq = {}
    for word in tokens:
        freq[word] = freq.get(word, 0.0) + 1.0
    for word in freq:
        freq[word] = freq[word]/len(tokens)
    return freq


def get_top_n_words(freq_dict: dict[str, float], top_n: int) -> Sequence[str] | None:
    """
    Finds the most common words

    Args:
        freq_dict (dict[str, float]): Dictionary with frequencies
        top_n (int): Number of the most common words

    Returns:
        Sequence[str] | None: Sequence of the most common words.
        Returns None in case of incorrect input types or non-positive top_n.
    """
    if not isinstance(freq_dict, dict) or not isinstance(top_n, int):
        return None
    if top_n <= 0:
        return None
    sorted_dict = sorted(freq_dict.items(), key = lambda x: (-x[1], x[0]))
    top_n_words = [word for word, _ in sorted_dict[:top_n]]
    return top_n_words



# Mark 6.


def create_language_profile(
    language: str, text: str, stop_words: Sequence[str]
) -> ProfileType | None:
    """
    Creates a language profile

    Args:
        language (str): Language name
        text (str): Text
        stop_words (Sequence[str]): Sequence of stop words (can be empty)

    Returns:
        ProfileType | None: Language profile.
        Returns None in case of incorrect input types.
    """
    if not all((
        isinstance(language, str),
        isinstance(text, str),
        isinstance(stop_words, (list, tuple)),
    )):
        return None
    for word in stop_words:
        if not isinstance(word, str):
            return None
    prepared_text = tokenize(text)
    if prepared_text is None:
        return None
    without_stop_words = remove_stop_words(prepared_text, stop_words)
    if without_stop_words is None:
        return None
    words_frequencies = calculate_frequencies(without_stop_words)
    if words_frequencies is None:
        return None
    lang_profile = (language, words_frequencies, len(words_frequencies.keys()))
    return lang_profile



def check_profile(profile: ProfileType) -> bool:
    """
    Checks profile structure

    Args:
        profile (ProfileType): Profile to check

    Returns:
        bool: Returns True if the profile has right structure and types,
        otherwise returns False.
    """
    if not isinstance(profile, tuple) or len(profile) != 3:
        return False
    if not isinstance(profile[0], str) or not isinstance(profile[1], dict):
        return False
    for keys, values in profile[1].items():
        if not isinstance(keys, str) or not isinstance(values, float):
            return False
    if not isinstance(profile[2], int):
        return False
    return True

def compare_profiles_by_top_n(
    unknown_profile: ProfileType, profile_to_compare: ProfileType, top_n: int
) -> float | None:
    """
    Compares profiles and calculates the distance using top n words

    Args:
        unknown_profile (ProfileType): Unknown profile
        profile_to_compare (ProfileType): Profile of a known language
        top_n (int): Number of the most common words
    Returns:
        float | None: The distance between profiles.
        Returns None in case of incorrect input types.
    """
    if not all((
        check_profile(unknown_profile),
        check_profile(profile_to_compare),
        isinstance(top_n, int),
    )):
        return None
    if top_n <= 0:
        return None
    sorted_dict_unkn = get_top_n_words(unknown_profile[1], top_n)
    sorted_dict_comp = get_top_n_words(profile_to_compare[1], top_n)
    if sorted_dict_unkn is None or sorted_dict_comp is None:
        return None
    compared_words = [value for value in sorted_dict_unkn if value in sorted_dict_comp]
    return len(compared_words)/top_n



def detect_language_by_top_n(
    unknown_profile: ProfileType, profile_1: ProfileType, profile_2: ProfileType, top_n: int
) -> str | None:
    """
    Detects the language of an unknown profile

    Args:
        unknown_profile (ProfileType): Unknown profile
        profile_1 (ProfileType): Profile for comparison
        profile_2 (ProfileType): Another profile for comparison
        top_n (int): Number of the most common words

    Returns:
        str | None: Unknown profile language.
        Returns None in case of incorrect input types.
    """
    if not all((
        check_profile(unknown_profile),
        check_profile(profile_1),
        check_profile(profile_2),
        isinstance(top_n, int),
    )):
        return None
    first_compared = compare_profiles_by_top_n(unknown_profile, profile_1, top_n)
    second_compared = compare_profiles_by_top_n(unknown_profile, profile_2, top_n)
    if first_compared is None or second_compared is None:
        return None
    if first_compared > second_compared:
        return profile_1[0]
    if first_compared < second_compared:
        return profile_2[0]
    return min(profile_1[0], profile_2[0])


# Mark 8


def calculate_mse(predicted: Sequence[float], actual: Sequence[float]) -> float | None:
    """
    Calculates mean squared error between predicted and actual values.

    Args:
        predicted (Sequence[float]): Sequence of predicted values
        actual (Sequence[float]): Sequence of actual values

    Returns:
        float | None: The score
        Returns None in case of incorrect input types or mismatched length.
        In case of empty inputs, returns 0.0.
    """
    if not isinstance(predicted, (list, tuple)) or not isinstance(actual, (list, tuple)):
        return None
    if not predicted or not actual:
        return 0.0
    if len(actual) != len(predicted):
        return None
    for n in predicted:
        if not isinstance(n, float):
            return None
    for n in actual:
        if not isinstance(n, float):
            return None
    mse = 0
    for i, a in enumerate(actual):
        difference = a - predicted[i]
        mse += difference ** 2
    return mse / len(actual)



def compare_profiles_by_mse(
    unknown_profile: ProfileType, profile_to_compare: ProfileType
) -> float | None:
    """
    Compares two language profiles using the MSE metric.

    Args:
        unknown_profile (ProfileType): Unknown profile
        profile_to_compare (ProfileType): Profile
            to compare the unknown profile with

    Returns:
        float | None: The distance between the profiles.
        In case of corrupt input arguments or invalid profile structure, None is returned.
    """
    if not all((
        check_profile(unknown_profile),
        check_profile(profile_to_compare),
    )):
        return None
    all_words = set(unknown_profile[1].keys()) | set(profile_to_compare[1].keys())
    actual = []
    predicted = []
    for item in all_words:
        actual.append(unknown_profile[1].get(item, 0.0))
        predicted.append(profile_to_compare[1].get(item, 0.0))
    return calculate_mse(predicted, actual)



def detect_language_by_mse(
    unknown_profile: ProfileType, profile_1: ProfileType, profile_2: ProfileType
) -> str | None:
    """
    Detects the language of an unknown profile.

    Args:
        unknown_profile (ProfileType): Profile
            to determine the language of
        profile_1 (ProfileType): Known profile
        profile_2 (ProfileType): Another known profile

    Returns:
        str | None: Unknown profile language.
        Returns None in case of incorrect input types.
    """
    if not all((
        check_profile(unknown_profile),
        check_profile(profile_1),
        check_profile(profile_2),
    )):
        return None
    compared_with_first = compare_profiles_by_mse(unknown_profile, profile_1)
    compared_with_second = compare_profiles_by_mse(unknown_profile, profile_2)
    if compared_with_first is None:
        return None
    if compared_with_second is None:
        return None
    if compared_with_first < compared_with_second:
        return profile_1[0]
    if compared_with_second < compared_with_first:
        return profile_2[0]
    return min(profile_1[0], profile_2[0])



# Mark 10
def save_profile(profile: ProfileType, save_path: str) -> bool:
    """
    Saves a language profile

    Args:
        profile (ProfileType): Profile
        save_path (str): Path to the folder to save profile

    Returns:
        bool: False in case of incorrect input types or if the profile
        is missing obligatory keys. True if the profile is saved.
    """
    # if not all((check_profile(profile), isinstance(save_path, str))):
    #     return False
    # prepared_profile = {'name': profile[0], 'freq': profile[1], 'n_words': profile[2]}
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(prepared_profile, f, ensure_ascii=False, indent=4)
    # return True



def load_profile(path_to_file: str) -> ProfileType | None:
    """
    Loads a language profile.

    Args:
        path_to_file (str): Path to the language profile

    Returns:
        ProfileType | None: Loaded profile.
        Returns None in case of incorrect input types.
    """
    # if not isinstance(path_to_file, str):
    #     return None
    # with open(path_to_file, 'r', encoding='utf-8') as f:
    #     profile = json.load(f)
    # if not isinstance(profile, dict):
    #     return None
    # processed_profile = (profile.get('name'), profile.get('freq'), profile.get('n_words'))
    # if not check_profile(processed_profile):
    #     return None
    # return processed_profile



def collect_profiles(paths_to_profiles: Sequence[str]) -> Sequence[ProfileType] | None:
    """
    Collects profiles for a given path.

    Args:
        paths_to_profiles (Sequence[str]): Sequence of paths to the profiles

    Returns:
        Sequence[ProfileType] | None: Sequence of loaded profiles.
        Returns None in case of incorrect input types.
    """
    # if not isinstance(paths_to_profiles, (tuple, list)):
    #     return None
    # for path in paths_to_profiles:
    #     if not isinstance(path, str):
    #         return None
    # downloaded_profiles = [load_profile(item) for item in paths_to_profiles]
    # for x in downloaded_profiles:
    #     if not check_profile(x):
    #         return None
    # return downloaded_profiles



def detect_language_advanced(
    unknown_profile: ProfileType, known_profiles: Sequence[ProfileType], top_n: int
) -> Sequence[tuple[str, dict[str, float]]] | None:
    """
    Detects the language of an unknown profile.

    Args:
        unknown_profile (ProfileType): Profile
            to determine the language of
        known_profiles (Sequence[ProfileType]): Known profiles
        top_n (int): Number of popular words

    Returns:
        Sequence[tuple[str, dict[str, float]]] | None: Sorted sequence of tuples
        containing a language and a distance via both metrics.
        The sequence is sorted by best MSE value, then by best Top-N value.
        Returns None in case of incorrect input types.
    """
    # if not all((
    #     check_profile(unknown_profile),
    #     isinstance(known_profiles, (list, tuple)),
    #     isinstance(top_n, int),
    # )):
    #     return None
    # for profile in known_profiles:
    #     if not check_profile(profile):
    #         return None
    # profiles = []
    # for profile in known_profiles:
    #     by_mse = compare_profiles_by_mse(unknown_profile, profile)
    #     by_top_n = compare_profiles_by_top_n(unknown_profile, profile, top_n)
    #     profiles.append((profile[0], {'MSE': by_mse, 'Top-N': by_top_n}))
    # sorted_profiles = sorted(profiles, key=lambda x: (x[1].get('MSE'), -x[1].get('Top-N'), x[0]))
    # return sorted_profiles



def print_report(
    unknown_profile: ProfileType, metrics_stats: Sequence[tuple[str, dict[str, float]]], top_n: int
) -> None:
    """
    Prints report for detection of language.

    Args:
        unknown_profile (ProfileType): Profile
        metrics_stats (Sequence[tuple[str, dict[str, float]]]): Sequence with distances for
            available language comparison and metrics
        top_n (int): Number of popular words

    In case of incorrect type inputs, does not print anything.
    """
    # if not all((
    #     check_profile(unknown_profile),
    #     isinstance(metrics_stats, (list, tuple)),
    #     isinstance(top_n, int),
    # )):
    #     return None
    # print('Unknown language stats')
    # print('=' * 22)
    # print(f'Popular words: {get_top_n_words(unknown_profile[1], top_n)}')
    # print(f"Max length word: '{max(unknown_profile[1].keys(), key = len)}'")
    # print(f"Min length word: '{min(unknown_profile[1].keys(), key = len)}'")

