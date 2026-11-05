"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
)

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open(
        "./lab_3_generate_by_ngrams/assets/Harry_Potter.txt", "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()

    processor = TextProcessor('_')

    encoded = processor.encode(text)
    assert encoded, 'Encoding failed'
    print(encoded)

    decoded = processor.decode(encoded)
    assert decoded, 'Decoding failed'
    print(decoded)

    model = NGramLanguageModel(encoded, 3)

    built = model.build()
    assert built, 'Building failed'
    print(built)

    generator = GreedyTextGenerator(model, processor)

    prediction = generator.run('Vernon', 51)
    assert prediction, 'Predicting failed'
    print(prediction)


if __name__ == "__main__":
    main()
