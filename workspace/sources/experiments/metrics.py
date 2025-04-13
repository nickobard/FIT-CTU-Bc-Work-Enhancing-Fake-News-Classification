from abc import ABC


class Metric(ABC):
    pass


class FalsePositiveRate(Metric):
    def __init__(self):
        self.name = 'false_positive_rate'
        self.greater_is_better = False
