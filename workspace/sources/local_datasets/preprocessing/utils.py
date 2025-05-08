from .tokenization import BertBaseUncasedTokenizer, NLTKTokenizer
from .encoding import BertBaseUncasedEncoder, HuggingFaceDatasetConversion
from .cleaning import NoiseReduction, Stemming, Lemmatization


class PreprocessingPipeline(list):
    def __init__(self, name, iterable=()):
        super().__init__(iterable)
        self.name = name

    def __repr__(self):
        return f"<PreprocessingPipeline {self.name!r}: {list(self)}>"


minimal_bert_pipeline = PreprocessingPipeline(name='minima_bert_pipeline',
                                              iterable=[BertBaseUncasedTokenizer(),
                                                        BertBaseUncasedEncoder(),
                                                        HuggingFaceDatasetConversion()])

cleaned_bert_pipeline = PreprocessingPipeline(
    name='noise_reduction_bert_pipeline',
    iterable=[NoiseReduction(),
              BertBaseUncasedTokenizer(),
              BertBaseUncasedEncoder(),
              HuggingFaceDatasetConversion()])

cleaned_stemmed_bert_pipeline = PreprocessingPipeline(name='noise_reduction_with_stemming_bert_pipeline',
                                                      iterable=[NoiseReduction(),
                                                                NLTKTokenizer(),
                                                                Stemming(),
                                                                BertBaseUncasedTokenizer(),
                                                                BertBaseUncasedEncoder(),
                                                                HuggingFaceDatasetConversion()])

cleaned_lemmatized_bert_pipeline = PreprocessingPipeline(name='noise_reduction_with_lemmatizing_bert_pipeline',
                                                         iterable=[NoiseReduction(),
                                                                   NLTKTokenizer(),
                                                                   Lemmatization(),
                                                                   BertBaseUncasedTokenizer(),
                                                                   BertBaseUncasedEncoder(),
                                                                   HuggingFaceDatasetConversion()])
