from .bert_base_uncased import BertBaseUncasedEncoder


class RobertaEncoder(BertBaseUncasedEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_name = 'roberta-base'
