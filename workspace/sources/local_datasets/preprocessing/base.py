from abc import ABC, abstractmethod


class Preprocessing(ABC):

    @abstractmethod
    def preprocess(self, data):
        pass

class BertTokenizer(Preprocessing):
    def preprocess(self, data):
        raise NotImplementedError
