from transformers import BertTokenizer
from .base import Preprocessing


class BertBasedUncasedTokenizer(Preprocessing):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def hyperparameters_output(self):
        return {'tokenizer': 'bert-base-uncased'}

    def preprocess(self, data):
        tokenized_features = data.features.apply(self.tokenizer.tokenize)
        data.features = tokenized_features
        return data
