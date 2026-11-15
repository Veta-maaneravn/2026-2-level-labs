"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
    WordProcessor,
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
    assert encoded, 'Encoding for letters failed'
    print('Encoded text (first 1000 symbols):', encoded[:1000])

    decoded = processor.decode(encoded)
    assert decoded, 'Decoding for letters failed'
    print('Decoded text (first 1000 symbols):', decoded[:1000])

    model = NGramLanguageModel(encoded, 3)

    built = model.build()
    assert built, 'Model building for letters failed'
    print('Are frequencies loaded:', built)

    generator = GreedyTextGenerator(model, processor)

    prediction = generator.run('Vernon', 51)
    assert prediction, 'Greedy generation for letters failed'
    print('The work of GreedyTextGenerator:', prediction)

    beam_searcher = BeamSearchTextGenerator(model, processor, 7).run('Vernon', 56)

    assert beam_searcher, 'BeamSearch generation for letters failed'
    print('The work of BeamSearchTextGenerator:', beam_searcher)

    model_reader = NGramLanguageModelReader(
        'lab_3_generate_by_ngrams/assets/en_own.json', '_'
    )

    ngram_2 = model_reader.load(2)
    ngram_3 = model_reader.load(3)
    ngram_4 = model_reader.load(4)
    assert (
        ngram_2 is not None
        and ngram_3 is not None
        and ngram_4 is not None
    ), 'Loading models failed'

    text_processor = model_reader.get_text_processor()

    back_off = BackOffGenerator((ngram_2, ngram_3, ngram_4), text_processor).run('Vernon', 100)

    assert back_off, 'BackOffGenerator for letters failed'
    print('The work of BackOffGenerator:', back_off)

    result = back_off
    assert result

    processor_text = WordProcessor('<eos>')

    encoded_text = processor_text.encode(text)
    assert encoded_text, 'Encoding for words failed'
    print('Encoded text (first 1000 words):', encoded_text[:1000])

    ngram_text_2, ngram_text_3, ngram_text_4 = (
        NGramLanguageModel(encoded_text, n) for n in (2, 3, 4)
    )
    assert all(n.build() for n in (ngram_text_2, ngram_text_3, ngram_text_4)), 'Building failed'

    generator_text = GreedyTextGenerator(ngram_text_4, processor_text).run('Harry looked at', 12)
    assert generator_text, 'Greedy generation for words failed'
    print('The work of GreedyTextGenerator:', generator_text)

    beam_searcher_text = BeamSearchTextGenerator(
        ngram_text_4, processor_text, 3
    ).run('Harry looked at', 12)
    assert beam_searcher_text, 'Beam Search for words failed'
    print('The work of Beam Search:', beam_searcher_text)

    back_off_text = BackOffGenerator(
        (ngram_text_2, ngram_text_3, ngram_text_4),
        processor_text
    ).run('Harry looked at', 12)
    assert back_off_text, 'BackOff for words failed'
    print('The work of BackOff:', back_off_text)

    generator_letters = generator.run('Harry looked at', 60)
    assert generator_letters, 'Generating by letters failed'
    print(
        'For comparison: GreedyTextGenerator with letters:',
        generator_letters
    )


if __name__ == "__main__":
    main()
