from .bert_base_uncased import BertBaseUncased


class Roberta(BertBaseUncased):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'roberta-base'
