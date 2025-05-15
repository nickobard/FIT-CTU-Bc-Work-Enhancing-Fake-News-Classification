from .base import Transformer


class BERT(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'BERT'
        self.transformers_identifier = 'bert-base-uncased'


class DistilliBERT(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'DistilliBERT'
        self.transformers_identifier = 'distilbert-base-uncased'


class RoBERTa(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'RoBERTa'
        self.transformers_identifier = 'roberta-base'
