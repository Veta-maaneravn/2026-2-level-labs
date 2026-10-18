"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable
from lab_2_tokenize_by_bpe.main import (
    collect_frequencies,
    train,
    get_vocabulary,
    decode,
    load_vocabulary,
    encode,
)


def main() -> None:
    """
    Launches an implementation
    """
    result = None

    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open('lab_2_tokenize_by_bpe/assets/ru_raw.txt', 'r', encoding='utf-8') as text_file:
        text_to_translate = text_file.read()
    with open('lab_2_tokenize_by_bpe/assets/ru_encoded.txt', 'r', encoding='utf-8') as text_file:
        ideal_translation = [int(value) for value in text_file.read().split()]

    with open('lab_2_tokenize_by_bpe/assets/ru_raw.txt', 'r', encoding='utf-8') as text_file:
        text_to_translate = text_file.read()

    with open('lab_2_tokenize_by_bpe/assets/ru_encoded.txt', 'r', encoding='utf-8') as text_file:
        ideal_translation = [int(value) for value in text_file.read().split()]

    secret = 'lab_2_tokenize_by_bpe/assets/secrets/secret_3.txt'
    with open(secret, 'r', encoding='utf-8') as text_file:
        secret_text = text_file.read()

    encoded_secret = []
    for value in secret_text.split():
        if value.lstrip('-').isdigit():
            encoded_secret.append(int(value))

    collected_frequencies = collect_frequencies(text, None, '</s>')
    assert collected_frequencies, 'Collecting failed'

    trained = train(collected_frequencies, 100)
    assert trained, 'Training failed'

    vocabulary = get_vocabulary(trained, '<unk>')
    assert vocabulary, 'Getting vocabulary failed'

    decoded = decode(secret_text, vocabulary, '</s>')
    assert decoded, 'Decoding failed'
    print('Dedcoded text', decoded)

    loaded = load_vocabulary('lab_2_tokenize_by_bpe/assets/vocab.json')
    assert loaded, 'Loading failed'

    encoded = encode(text_to_translate, loaded, '\u2581', None, '<unk>')
    assert encoded, 'Encoding failed'

    compare_len = min(20, len(encoded), len(ideal_translation))
    is_match = all(encoded[i] == ideal_translation[i] for i in range(compare_len))
    print('Compare result:', is_match)

    result = encoded
    assert result, "Translation not working"


if __name__ == "__main__":
    main()
