from .bert_base_uncased import BertBaseUncasedEncoder


class DistillibertEncoder(BertBaseUncasedEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_name = 'distilbert-base-uncased'
