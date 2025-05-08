from .base import HuggingFaceDatasetConversion
from .tokenization import BertBaseUncasedTokenizer, NLTKTokenizer
from .encoding import BertBaseUncasedEncoder
from .cleaning import NoiseReduction, Stemming, Lemmatization

minimal_bert_pipeline = [BertBaseUncasedTokenizer(),
                         BertBaseUncasedEncoder(),
                         HuggingFaceDatasetConversion()]

cleaned_bert_pipeline = [NoiseReduction(),
                         BertBaseUncasedTokenizer(),
                         BertBaseUncasedEncoder(),
                         HuggingFaceDatasetConversion()]

cleaned_stemmed_bert_pipeline = [NoiseReduction(),
                                 NLTKTokenizer(),
                                 Stemming(),
                                 BertBaseUncasedTokenizer(),
                                 BertBaseUncasedEncoder(),
                                 HuggingFaceDatasetConversion()]

cleaned_lemmatized_bert_pipeline = [NoiseReduction(),
                                    NLTKTokenizer(),
                                    Lemmatization(),
                                    BertBaseUncasedTokenizer(),
                                    BertBaseUncasedEncoder(),
                                    HuggingFaceDatasetConversion()]
