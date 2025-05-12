from .bert_base_uncased import BertBaseUncased


class Distillibert(BertBaseUncased):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'distilbert-base-uncased'
