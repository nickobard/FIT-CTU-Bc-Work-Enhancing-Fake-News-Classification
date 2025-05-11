from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput
from contextlib import contextmanager

from ..utils import create_and_get_local_logger


class TrainerWithEmbeddingsCollection(Trainer):
    def __init__(self,
                 logger=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger if logger else create_and_get_local_logger(self.__class__.__name__)
        self.embeddings_collection = None
        self.output_hidden_states = False

    @contextmanager
    def collect_hidden_states(self):
        # enter: turn on hidden-state collection
        self.output_hidden_states = True
        self.embeddings_collection = []
        try:
            yield self
        finally:
            # exit: turn it off
            self.output_hidden_states = False
            self.embeddings_collection = None

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs['output_hidden_states'] = self.output_hidden_states
        loss, outputs, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if not self.output_hidden_states:
            return loss, outputs, labels
        logits, hidden_states = outputs
        last_hidden_state = hidden_states[-1][:, 0, :].cpu().numpy()
        self.embeddings_collection.append(last_hidden_state)
        return loss, logits, labels

    def get_collected_embeddings(self):
        return np.vstack(self.embeddings_collection)
