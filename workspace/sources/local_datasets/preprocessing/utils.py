from .tokenization import BertBaseUncasedTokenizer, NLTKTokenizer
from .encoding import BertBaseUncasedEncoder
from .preprocessings import NoiseReduction, Stemming, Lemmatization

minimal_bert_pipeline = [BertBaseUncasedTokenizer(), BertBaseUncasedEncoder()]
cleaned_bert_pipeline = [NoiseReduction(), BertBaseUncasedTokenizer(), BertBaseUncasedEncoder()]
cleaned_stemmed_bert_pipeline = [NoiseReduction(), NLTKTokenizer(), Stemming(), BertBaseUncasedTokenizer(),
                                 BertBaseUncasedEncoder()]
cleaned_lemmatized_bert_pipeline = [NoiseReduction(), NLTKTokenizer(), Lemmatization(), BertBaseUncasedTokenizer(),
                                    BertBaseUncasedEncoder()]
