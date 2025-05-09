from logging import getLogger

from .tokenization import BertBaseUncasedTokenizer, NLTKTokenizer
from .encoding import BertBaseUncasedEncoder, HuggingFaceDatasetConversion
from .cleaning import NoiseReduction, Stemming, Lemmatization
from ...utils import log_params


class PreprocessingPipeline(list):
    def __init__(self, name, iterable=()):
        super().__init__(iterable)
        self.name = name
        self.logger = None

    def init(self, logger=None):
        self._set_logger(logger)
        self.logger.info(f'Initializing preprocessing pipeline: {repr(self)}')
        for preprocessing in self:
            preprocessing.init(logger=self.logger)
        return self

    def log_params(self, logger=None):
        self._set_logger(logger)
        params = {
            'preprocessing_pipeline_name': self.name,
            'preprocessing_pipeline_representation': repr(self),
            'preprocessing_pipeline': [p.name() for p in self],
        }
        log_params(params,
                   logger=self.logger)
        for preprocessing in self:
            preprocessing.log_params(logger=self.logger)
        return params

    def _set_logger(self, logger):
        if logger:
            self.logger = logger
        else:
            self.logger = self.logger if self.logger else getLogger()
        return self

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
