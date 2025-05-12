from .tokenization import NLTKTokenizer
from .encoders.bert_base_uncased import BertBaseUncasedEncoder
from .cleaning import NoiseReduction, Stemming, Lemmatization
from ...utils import log_params, create_and_get_local_logger


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

    def is_empty(self):
        return len(self) == 0

    def _set_logger(self, logger):
        if logger:
            self.logger = logger
        else:
            self.logger = self.logger if self.logger else create_and_get_local_logger(self.__class__.__name__)
        return self

    def assemble_signature(self):
        class_name = self.name
        params_signature = [p.assemble_signature() for p in self]
        signature = f'pipeline={class_name}({params_signature})'
        return signature

    def __repr__(self):
        return f"<PreprocessingPipeline {self.name!r}: {list(self)}>"


minimal_bert_pipeline = PreprocessingPipeline(name='minima_bert_pipeline',
                                              iterable=[BertBaseUncasedEncoder()])

cleaned_bert_pipeline = PreprocessingPipeline(
    name='noise_reduction_bert_pipeline',
    iterable=[NoiseReduction(),
              BertBaseUncasedEncoder()])

cleaned_stemmed_bert_pipeline = PreprocessingPipeline(name='noise_reduction_with_stemming_bert_pipeline',
                                                      iterable=[NoiseReduction(),
                                                                NLTKTokenizer(),
                                                                Stemming(),
                                                                BertBaseUncasedEncoder(is_split_into_words=True)
                                                                ])

cleaned_lemmatized_bert_pipeline = PreprocessingPipeline(name='noise_reduction_with_lemmatizing_bert_pipeline',
                                                         iterable=[NoiseReduction(),
                                                                   NLTKTokenizer(),
                                                                   Lemmatization(),
                                                                   BertBaseUncasedEncoder(is_split_into_words=True)
                                                                   ])
