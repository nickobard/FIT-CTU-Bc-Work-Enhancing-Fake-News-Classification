from models.transformers_models import TransformersModels
from scipy.special import softmax
import mlflow
import datasets as hf_datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, set_seed
from models.callbacks import HF_CustomMLflowCallback
import os
import glob

from experiments.metrics import compute_standard_metrics, FalsePositiveRate


class BertBaseUncased(TransformersModels):

    def __init__(self, training_arguments=None, main_metric=FalsePositiveRate()):
        super().__init__(main_metric)
        self.name = "bert-base-uncased"
        training_arguments = {} if training_arguments is None else training_arguments
        self.training_args = {**self.get_default_training_args(), **training_arguments}
        self.output_dir = None
        self.logging_dir = None

    def has_checkpoints(self):
        """Check if there are checkpoints available for model training resumption."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return False
        checkpoint_files = glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
        return len(checkpoint_files) > 0

    def init(self, logger=None, random_state=None):
        super().init(logger, random_state)
        mlflow.log_param('model_name', self.name)
        self.output_dir = mlflow.active_run().data.params.get('output_dir',
                                                              os.path.join(self.get_artifacts_path(),
                                                                           'checkpoints'))
        self.logging_dir = mlflow.active_run().data.params.get('logging_dir',
                                                               os.path.join(self.get_artifacts_path(), 'logs'))
        return self

    @classmethod
    def get_default_training_args(cls):
        return {
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 5e-05,
            'weight_decay': 0.01,
        }

    def fit(self, dataset):
        self.dataset = dataset
        set_seed(self.random_state)
        self.model = BertForSequenceClassification.from_pretrained(self.name, num_labels=2)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            per_device_train_batch_size=self.training_args['batch_size'],
            per_device_eval_batch_size=self.training_args['batch_size'],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            save_steps=100,
            weight_decay=self.training_args['weight_decay'],
            learning_rate=self.training_args['learning_rate'],
            load_best_model_at_end=True,
            metric_for_best_model=self.main_metric.name,
            greater_is_better=self.main_metric.greater_is_better,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset.preprocessed_train_set.dataset,
            eval_dataset=dataset.preprocessed_train_set.dataset,
            compute_metrics=compute_standard_metrics,
            callbacks=[HF_CustomMLflowCallback()]
        )
        self.trainer.train(resume_from_checkpoint=True if self.has_checkpoints() else False)

    def evaluate(self):
        # Extract predictions and labels
        logits, labels, metrics = self.trainer.predict(self.dataset.preprocessed_test_set.dataset,
                                                       metric_key_prefix='test')
        best_epoch = self.trainer.state.epoch
        mlflow.log_metrics({f"best_{key}": value for key, value in metrics.items()})
        mlflow.log_metric('best_epoch', best_epoch)

        probs = softmax(logits, axis=1)[:, 1]
        predictions = (probs > 0.5).astype(int)

        mlflow.log_params({
            'evaluation_logits': logits,
            'evaluation_probabilities': probs,
            'evaluations_predictions': predictions,
            'evaluation_labels': labels,
            'evaluation_metrics': metrics})

        return self
