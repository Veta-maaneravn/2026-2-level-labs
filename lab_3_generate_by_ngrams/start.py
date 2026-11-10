"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
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
    print('Encoded text:', encoded)

    decoded = processor.decode(encoded)
    assert decoded, 'Decoding failed'
    print('Decoded text:', decoded)

    model = NGramLanguageModel(encoded, 3)

    built = model.build()
    assert built, 'Building failed'
    print('Are frequencies loaded:', built)

    generator = GreedyTextGenerator(model, processor)

    prediction = generator.run('Vernon', 51)
    assert prediction, 'Predicting failed'
    print('The work of GreadyTextGenerator:', prediction)

    beam_searcher = BeamSearchTextGenerator(model, processor, 7)

    beam_searcher_generator = beam_searcher.run('Vernon', 56)
    assert beam_searcher_generator, 'Generation failed'
    print('The work of BeamSearchTextGenerator:', beam_searcher_generator)


if __name__ == "__main__":
    main()
