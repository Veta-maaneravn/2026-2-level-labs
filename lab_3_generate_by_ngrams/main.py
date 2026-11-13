"""
Lab 3.

Beam-search and natural language generation evaluation
"""
# pylint:disable=too-few-public-methods
import json
import math


class TextProcessor:
    """
    Handle text tokenization, encoding and decoding.

    Attributes:
        _end_of_word_token (str): A token denoting word boundary
        _storage (dict): Dictionary in the form of <token: identifier>
    """

    def __init__(self, end_of_word_token: str) -> None:
        """
        Initialize an instance of TextProcessor.

        Args:
            end_of_word_token (str): A token denoting word boundary
        """
        self._end_of_word_token = end_of_word_token
        self._storage = {end_of_word_token: 0}


    def _tokenize(self, text: str) -> tuple[str, ...] | None:
        """
        Tokenize text into unigrams, separating words with special token.

        Punctuation and digits are removed. EoW token is appended after the last word in two cases:
        1. It is followed by punctuation
        2. It is followed by space symbol

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...] | None: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str):
            return None
        cleaned_parts = []
        for item in text.lower():
            if item.isalpha():
                cleaned_parts.extend(item)
            if item.isspace():
                if cleaned_parts[-1] != self._end_of_word_token:
                    cleaned_parts.append(self._end_of_word_token)
        if not cleaned_parts:
                return None
        if text[-1] in '.?!':
                cleaned_parts.append(self._end_of_word_token)
        return tuple(cleaned_parts)


    def get_id(self, element: str) -> int | None:
        """
        Retrieve a unique identifier of an element.

        Args:
            element (str): String element to retrieve identifier for

        Returns:
            int | None: Integer identifier that corresponds to the given element

        In case of corrupt input arguments or arguments not included in storage,
        None is returned
        """
        if not isinstance(element, str):
            return None
        return self._storage.get(element)


    def get_end_of_word_token(self) -> str:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._end_of_word_token attribute.

        Returns:
            str: EoW token
        """
        return self._end_of_word_token


    def get_token(self, element_id: int) -> str | None:
        """
        Retrieve an element by unique identifier.

        Args:
            element_id (int): Identifier to retrieve identifier for

        Returns:
            str | None: Element that corresponds to the given identifier

        In case of corrupt input arguments or arguments not included in storage, None is returned
        """
        if not isinstance(element_id, int):
            return None
        for key, value in self._storage.items():
            if value == element_id:
                return key
        return None


    def encode(self, text: str) -> tuple[int, ...] | None:
        """
        Encode text.

        Tokenize text, assign each symbol an integer identifier and
        replace letters with their ids.

        Args:
            text (str): An original text to be encoded

        Returns:
            tuple[int, ...] | None: Processed text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str):
            return None
        tokens = self._tokenize(text)
        if not tokens:
            return None
        for item in tokens:
            self._put(item)
        encoded_corpus = []
        for token in tokens:
            token_id = self.get_id(token)
            if token_id is None:
                return None
            encoded_corpus.append(token_id)
        if not encoded_corpus:
            return None
        return tuple(encoded_corpus)


    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str):
            return
        if len(element) != 1:
            return
        if element not in self._storage:
            self._storage[element] = len(self._storage)


    def decode(self, encoded_corpus: tuple[int, ...]) -> str | None:
        """
        Decode and postprocess encoded corpus by converting integer identifiers to string.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(encoded_corpus, tuple):
            return None
        for i in encoded_corpus:
            if not isinstance(i, int):
                return None
        first_decoded = self._decode(encoded_corpus)
        if first_decoded is None:
            return None
        postprocessed = self._postprocess_decoded_text(first_decoded)
        if postprocessed is None:
            return None
        return postprocessed


    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict):
            return
        actual_dict = content.get('freq')
        if not isinstance(actual_dict, dict):
            return
        for key in actual_dict:
            for letter in key:
                if letter.isalpha():
                    self._put(letter.lower())


    def _decode(self, corpus: tuple[int, ...]) -> tuple[str, ...] | None:
        """
        Decode sentence by replacing ids with corresponding letters.

        Args:
            corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[str, ...] | None: Sequence with decoded tokens

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(corpus, tuple):
            return None
        for i in corpus:
            if not isinstance(i, int):
                return None
        decoded_corpus = []
        for item in corpus:
            decoded = self.get_token(item)
            if decoded is None:
                return None
            decoded_corpus.append(decoded)
        if not decoded_corpus:
            return None
        return tuple(decoded_corpus)


    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str | None:
        """
        Convert decoded sentence into the string sequence.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(decoded_corpus, tuple):
            return None
        for i in decoded_corpus:
            if not isinstance(i, str):
                return None
        joined_corpus = ''.join(decoded_corpus).replace(self._end_of_word_token, ' ')
        cleaned_text = ' '.join(joined_corpus.split())
        if not cleaned_text:
            return None
        return cleaned_text.capitalize() + '.'


class WordProcessor(TextProcessor):
    """
    Handle text tokenization, encoding and decoding on the level of words.

    Attributes:
        _end_of_word_token (str): A token denoting sentence boundary
        _storage (dict): Dictionary in the form of <token: identifier>
    """
    def __init__(self, end_of_word_token):
        super().__init__('<eos>')


    def _tokenize(self, text: str) -> tuple[str, ...] | None:
        """
        Tokenize text into words, separating sentences with special token.

        Punctuation and digits are removed. Special token is appended after the word
        that ends a sentence, i.e. is followed by '.', '!' or '?'.

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...] | None: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case text does not contain any words, None is returned.
        """
        if not isinstance(text, str):
            return None
        tokenized_words = []
        for word in text.lower().split():
            cleaned_words = ''.join(
                symbol for symbol in word if symbol.isalpha()
            )
            if cleaned_words is None or not cleaned_words:
                continue
            tokenized_words.append(cleaned_words)
            if (word[-1] in '!?.' and
                tokenized_words[-1] != self._end_of_word_token):
                tokenized_words.append(self._end_of_word_token)
        if not tokenized_words:
            return None
        return tuple(tokenized_words)


    def _put(self, element: str) -> None:
        """
        Put a word into the storage, assign a unique id to it.

        Args:
            element (str): A word to put into storage

        In case of corrupt input arguments, an element is not added to storage
        """
        if (not isinstance(element, str) or
            not element):
            return None
        if element not in self._storage:
            self._storage[element] = len(self._storage)


    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str | None:
        """
        Convert decoded sequence of words into the string sequence.

        Words are separated with spaces, special tokens are replaced with full stops.
        Each sentence is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned.
        In case decoded corpus does not contain any words, None is returned.
        """
        if (not isinstance(decoded_corpus, tuple) or
            not decoded_corpus or
            not all(isinstance(i, str) for i in decoded_corpus)):
            return None
        joined_corpus = ' '.join(decoded_corpus)
        sentences = [
            s.strip() for s in joined_corpus.split(self._end_of_word_token)
        ]
        sentences = [s for s in sentences if s]
        if not sentences:
            return None
        prepared_sentences = [s.capitalize() for s in sentences]
        return '. '.join(prepared_sentences) + '.'


class NGramLanguageModel:
    """
    Store language model by n_grams, predict the next token.

    Attributes:
        __n_gram_size (int): A size of n-grams to use for language modelling
        _n_gram_frequencies (dict): Frequencies for n-grams
        _encoded_corpus (tuple): Encoded text
    """

    def __init__(self, encoded_corpus: tuple | None, n_gram_size: int) -> None:
        """
        Initialize an instance of NGramLanguageModel.

        Args:
            encoded_corpus (tuple | None): Encoded text
            n_gram_size (int): A size of n-grams to use for language modelling
        """
        self._encoded_corpus = encoded_corpus
        self.__n_gram_size = n_gram_size
        self._n_gram_frequencies = {}

    def get_n_gram_size(self) -> int:  # type: ignore[empty-body]
        """
        Retrieve value stored in self.__n_gram_size attribute.

        Returns:
            int: Size of stored n_grams
        """
        return self.__n_gram_size

    def set_n_grams(self, frequencies: dict) -> None:
        """
        Setter method for n-gram frequencies.

        Args:
            frequencies (dict): Computed in advance frequencies for n-grams
        """
        if not isinstance(frequencies, dict):
            return
        return self._n_gram_frequencies.update(frequencies)


    def build(self) -> bool:
        """
        Fill attribute `_n_gram_frequencies` from encoded corpus.

        Encoded corpus is stored in the attribute `_encoded_corpus`

        Returns:
            bool: True if attribute is filled successfully, otherwise False

        In case of corrupt input arguments or methods used return None,
        False is returned
        """
        if self._encoded_corpus is None:
            return False
        ngrams = self._extract_n_grams(self._encoded_corpus)
        if ngrams is None:
            return False
        ngram_counts = {}
        context_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            context = ngram[:-1]
            context_counts[context] = context_counts.get(context, 0) + 1
        self._n_gram_frequencies = {}
        for ngram, count in ngram_counts.items():
            context = ngram[:-1]
            self._n_gram_frequencies[ngram] = count / context_counts[context]
        return True


    def generate_next_token(self, sequence: tuple[int, ...]) -> dict | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of NGrams for continuation

        Returns:
            dict | None: Possible next tokens with their probabilities

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(sequence, tuple):
            return None
        for i in sequence:
            if not isinstance(i, int):
                return None
        if len(sequence) < self.__n_gram_size - 1:
            return None
        context = sequence[-(self.__n_gram_size - 1):]
        freq_tokens = {}
        for token, freq in self._n_gram_frequencies.items():
            if token[:-1] == context:
                freq_tokens[token[-1]] = freq
        return freq_tokens


    def _extract_n_grams(
        self, encoded_corpus: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...] | None:
        """
        Split encoded sequence into n-grams.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[tuple[int, ...], ...] | None: A tuple of extracted n-grams

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(encoded_corpus, tuple):
            return None
        for n in encoded_corpus:
            if not isinstance(n, int):
                return None
        extracted_ngrams = []
        for index in range(len(encoded_corpus) - self.__n_gram_size + 1):
            extracted_ngrams.append(encoded_corpus[index : index + self.__n_gram_size])
        if not extracted_ngrams:
            return None
        return tuple(extracted_ngrams)


class GreedyTextGenerator:
    """
    Greedy text generation by N-grams.

    Attributes:
        _model (NGramLanguageModel): A language model to use for text generation
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, language_model: NGramLanguageModel, text_processor: TextProcessor) -> None:
        """
        Initialize an instance of GreedyTextGenerator.

        Args:
            language_model (NGramLanguageModel): A language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._model = language_model
        self._text_processor = text_processor

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if not all((
            isinstance(prompt, str),
            isinstance(seq_len, int),
        )):
            return None
        if seq_len <= 0:
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        sequence = list(encoded_prompt)
        for _ in range(seq_len):
            next_token = self._model.generate_next_token(tuple(sequence))
            if not next_token:
                break
            max_token = max(next_token.items(), key=lambda x: (x[1], x[0]))[0]
            sequence.append(max_token)
        return self._text_processor.decode(tuple(sequence))


class BeamSearcher:
    """
    Beam Search algorithm for diverse text generation.

    Attributes:
        _beam_width (int): Number of candidates to consider at each step
        _model (NGramLanguageModel): A language model to use for next token prediction
    """

    def __init__(self, beam_width: int, language_model: NGramLanguageModel) -> None:
        """
        Initialize an instance of BeamSearcher.

        Args:
            beam_width (int): Number of candidates to consider at each step
            language_model (NGramLanguageModel): A language model to use for next token prediction
        """
        self._beam_width = beam_width
        self._model = language_model

    def get_next_token(self, sequence: tuple[int, ...]) -> list[tuple[int, float]] | None:
        """
        Retrieve candidate tokens for sequence continuation.

        Candidates are sorted by probability and then by token in descending order.
        Number of tokens retrieved must not be bigger than beam width parameter.

        The return value has the following format: [(token, probability), ...].

        Args:
            sequence (tuple[int, ...]): Base sequence to continue

        Returns:
            list[tuple[int, float]] | None: Tokens to use for base sequence continuation

        In case of corrupt input arguments or methods used return None.
        """
        if not isinstance(sequence, tuple):
            return None
        for i in sequence:
            if not isinstance(i, int):
                return None
        candidates = self._model.generate_next_token(sequence)
        if candidates is None:
            return None
        if not candidates:
            return []
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True
        )
        return sorted_candidates[:self._beam_width]


    def continue_sequence(
        self,
        sequence: tuple[int, ...],
        next_tokens: list[tuple[int, float]],
        sequence_candidates: dict[tuple[int, ...], float],
    ) -> dict[tuple[int, ...], float] | None:
        """
        Generate new sequences from the base sequence with next tokens provided.

        The base sequence is deleted after continued variations are added.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue
            next_tokens (list[tuple[int, float]]): Token for sequence continuation
            sequence_candidates (dict[tuple[int, ...], float]):
                Storage with all sequences generated

        Returns:
            dict[tuple[int, ...], float] | None: Updated sequence candidates

        In case of corrupt input arguments or unexpected behaviour of methods used return None.
        """
        if not all((
            isinstance(sequence, tuple),
            isinstance(next_tokens, list),
            isinstance(sequence_candidates, dict),
        )):
            return None
        if not (
            all(isinstance(n, int) for n in sequence) and
            all(isinstance(i, int) and
                isinstance(f, float) for i, f in next_tokens) and
            all(
                isinstance(k, tuple) and
                isinstance(v, float) and
                all(isinstance(s, int) for s in k)
                for k, v in sequence_candidates.items()
            )
        ):
            return None
        if (not next_tokens or
            len(next_tokens) > self._beam_width or
            sequence not in sequence_candidates
        ):
            return None
        current_penalty = sequence_candidates[sequence]
        for token, probability in next_tokens:
            new_token = sequence + (token,)
            new_penalty = current_penalty - math.log(probability)
            sequence_candidates[new_token] = new_penalty
        del sequence_candidates[sequence]
        return sequence_candidates


    def prune_sequence_candidates(
        self, sequence_candidates: dict[tuple[int, ...], float]
    ) -> dict[tuple[int, ...], float] | None:
        """
        Remove those sequence candidates that do not make top-N most probable sequences.

        Args:
            sequence_candidates (dict[tuple[int, ...], float]): Current candidate sequences

        Returns:
            dict[tuple[int, ...], float] | None: Pruned sequences

        In case of corrupt input arguments return None.
        """
        if (not isinstance(sequence_candidates, dict) or
            not sequence_candidates
        ):
            return None
        for k, v in sequence_candidates.items():
            if (not isinstance(k, tuple) or
                not isinstance(v, float)
            ):
                return None
            for i in k:
                if not isinstance(i, int):
                    return None
        sorted_sequence = sorted(
            sequence_candidates.items(),
            key=lambda x: (x[1], x[0])
        )
        return dict(sorted_sequence[:self._beam_width])


class BeamSearchTextGenerator:
    """
    Class for text generation with BeamSearch.

    Attributes:
        _language_model (NGramLanguageModel): Language model for next token prediction
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
        _beam_width (int): Beam width parameter for generation
        beam_searcher (BeamSearcher): Searcher instance for the language model
    """

    def __init__(
        self, language_model: NGramLanguageModel, text_processor: TextProcessor, beam_width: int
    ) -> None:
        """
        Initializes an instance of BeamSearchTextGenerator.

        Args:
            language_model (NGramLanguageModel): Language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
            beam_width (int): Beam width parameter for generation
        """
        self._language_model = language_model
        self._text_processor = text_processor
        self._beam_width = beam_width
        self.beam_searcher = BeamSearcher(beam_width, language_model)

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (not isinstance(prompt, str) or
            not isinstance(seq_len, int) or
            seq_len <= 0
        ):
            return None
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        candidates = {encoded_prompt: 0.0}
        for _ in range(seq_len):
            updated_candidates = {}
            for sequence, penalty in candidates.items():
                top_n = self._get_next_token(sequence)
                if top_n is None:
                    return None
                new_candidates = self.beam_searcher.continue_sequence(
                    sequence, top_n, {sequence: penalty}
                )
                if new_candidates is None:
                    updated_candidates[sequence] = penalty
                    continue
                updated_candidates.update(new_candidates)
            filtred = self.beam_searcher.prune_sequence_candidates(updated_candidates)
            if filtred is None:
                return None
            candidates = filtred
        min_penalty = min(
            candidates,
            key=lambda x: candidates[x]
        )
        return self._text_processor.decode(min_penalty)


    def _get_next_token(
        self, sequence_to_continue: tuple[int, ...]
    ) -> list[tuple[int, float]] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            list[tuple[int, float]] | None: Next tokens for sequence
            continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple):
            return None
        for i in sequence_to_continue:
            if not isinstance(i, int):
                return None
        return self.beam_searcher.get_next_token(sequence_to_continue)


class NGramLanguageModelReader:
    """
    Factory for loading language models ngrams from external JSON.

    Attributes:
        _json_path (str): Local path to assets file
        _eow_token (str): Special token for text processor
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, json_path: str, eow_token: str) -> None:
        """
        Initialize reader instance.

        Args:
            json_path (str): Local path to assets file
            eow_token (str): Special token for text processor
        """
        self._json_path = json_path
        self._eow_token = eow_token
        with open(self._json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self._content = data
        self._text_processor = TextProcessor(eow_token)
        self._text_processor.fill_from_ngrams(self._content)

    def load(self, n_gram_size: int) -> NGramLanguageModel | None:
        """
        Fill attribute `_n_gram_frequencies` from dictionary with N-grams.

        The N-grams taken from dictionary must be cleaned from digits and punctuation,
        their length must match n_gram_size, and spaces must be replaced with EoW token.

        Args:
            n_gram_size (int): Size of ngram

        Returns:
            NGramLanguageModel | None: Built language model.

        In case of corrupt input arguments or unexpected behaviour of methods used,
        None is returned.
        """
        if (not isinstance(n_gram_size, int) or
                n_gram_size < 2):
            return None
        json_dict = self._content.get('freq')
        if json_dict is None:
            return None
        ngrams = {}
        for keys, values in json_dict.items():
            prepared = ''.join(
                char.lower() if char.isalpha() else self._eow_token
                for char in keys
                if char.isalpha() or char.isspace()
            )
            if len(prepared) != n_gram_size:
                continue
            encoded = tuple(self._text_processor.get_id(t)
                            for t in prepared
            )
            if None in encoded:
                continue
            ngrams[encoded] = ngrams.get(encoded, 0) + values
        context_counts = {}
        for ngram, count in ngrams.items():
            context = ngram[:-1]
            context_counts[context] = context_counts.get(context, 0) + count
        freqs = {}
        for ngram, count in ngrams.items():
            context = ngram[:-1]
            if context_counts[context] > 0:
                freqs[ngram] = count / context_counts[context]
        exemplar = NGramLanguageModel(None, n_gram_size)
        exemplar.set_n_grams(freqs)
        return exemplar


    def get_text_processor(self) -> TextProcessor:  # type: ignore[empty-body]
        """
        Get method for the processor created for the current JSON file.

        Returns:
            TextProcessor: processor created for the current JSON file.
        """
        return self._text_processor


class BackOffGenerator:
    """
    Language model for back-off based text generation.

    Attributes:
        _language_models (dict[int, NGramLanguageModel]): Language models for next token prediction
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(
        self, language_models: tuple[NGramLanguageModel, ...], text_processor: TextProcessor
    ) -> None:
        """
        Initializes an instance of BackOffGenerator.

        Args:
            language_models (tuple[NGramLanguageModel, ...]):
                Language models to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._language_models = {}
        for item in language_models:
            self._language_models[item.get_n_gram_size()] = item
        self._text_processor = text_processor

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if (not isinstance(prompt, str) or
            not isinstance(seq_len, int) or
            seq_len <= 0
        ):
            return None
        if not prompt.endswith(self._text_processor.get_end_of_word_token()):
            prompt = prompt + self._text_processor.get_end_of_word_token()
        encoded_prompt = self._text_processor.encode(prompt)
        if encoded_prompt is None:
            return None
        result = list(encoded_prompt)
        for _ in range(seq_len):
            candidates = self._get_next_token(tuple(result))
            if candidates is None:
                return None
            if not candidates:
                break
            max_token = max(candidates.items(), key=lambda x: (x[1], x[0]))[0]
            result.append(max_token)
        return self._text_processor.decode(tuple(result))


    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple):
            return None
        for i in sequence_to_continue:
            if not isinstance(i, int):
                return None
        sorted_ngrams = sorted(self._language_models, reverse=True)
        for item in sorted_ngrams:
            nes_value = self._language_models.get(item)
            candidates = nes_value.generate_next_token(sequence_to_continue)
            if not candidates:
                continue
            if candidates:
                return candidates
        return None
