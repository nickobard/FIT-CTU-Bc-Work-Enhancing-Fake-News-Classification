import utils
from .transformers_models import TransformersModels
from scipy.special import softmax
import mlflow
from transformers import EarlyStoppingCallback, BertTokenizer, BertForSequenceClassification, Trainer, \
    TrainingArguments, set_seed
from .callbacks import HF_CustomMLflowCallback
import os
import glob
import pickle
from ..experiments.metrics import compute_standard_metrics, EvalLoss
from pathlib import Path
import json
from ..utils import log_params


class BertBaseUncased(TransformersModels):

    def __init__(self, training_arguments=None, main_metric=EvalLoss):
        super().__init__(main_metric)
        self.name = "bert-base-uncased"
        training_arguments = {} if training_arguments is None else training_arguments
        self.training_args = {**self.get_default_training_args(), **training_arguments}
        self.output_dir = None
        self.logging_dir = None
        self.evaluation_data = None
        self.dataset = None
        self.model = None
        self.trainer = None

    def has_checkpoints(self):
        """Check if there are checkpoints available for model training resumption."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return False
        checkpoint_files = glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
        return len(checkpoint_files) > 0

    def init(self, logger=None, random_state=None):
        super().init(logger, random_state)
        log_params({'model_name': self.name}, self.logger)
        log_params({'model_input_hyperparameters': self.training_args}, self.logger)
        log_params(
            {f'input_hyperparameter_{hparam_name}': value for hparam_name, value in self.training_args.items()},
            self.logger
        )
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
            'eval_batch_size': 8,
            'learning_rate': 5e-05,
            'weight_decay': 0.01,
            'early_stopping_patience': 3,
            'early_stopping_threshold': 0.001
        }

    def fit(self, dataset):
        self.dataset = dataset
        set_seed(self.random_state)
        self.model = BertForSequenceClassification.from_pretrained(self.name, num_labels=2)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            num_train_epochs=self.training_args['epochs'],
            per_device_train_batch_size=self.training_args['batch_size'],
            per_device_eval_batch_size=self.training_args['eval_batch_size'],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",  # regards only training
            logging_steps=10,  # regards only training
            weight_decay=self.training_args['weight_decay'],
            learning_rate=self.training_args['learning_rate'],
            load_best_model_at_end=True,
            metric_for_best_model=self.main_metric.name,
            greater_is_better=self.main_metric.greater_is_better,
        )

        callbacks = [HF_CustomMLflowCallback()]
        early_stopping_patience = self.training_args['early_stopping_patience']
        if early_stopping_patience is not None:
            early_stopping_threshold = self.training_args['early_stopping_threshold']
            esc = EarlyStoppingCallback(early_stopping_patience,
                                        early_stopping_threshold)
            callbacks.append(esc)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset.preprocessed_train_set.dataset,
            eval_dataset=dataset.preprocessed_train_set.dataset,
            compute_metrics=compute_standard_metrics,
            callbacks=callbacks
        )
        self.trainer.train(resume_from_checkpoint=True if self.has_checkpoints() else False)

    def evaluate(self):
        # Extract predictions and labels
        logits, labels, metrics = self.trainer.predict(self.dataset.preprocessed_test_set.dataset,
                                                       metric_key_prefix='test')
        best_epoch = self.trainer.state.epoch
        mlflow.log_metrics(metrics)
        mlflow.log_metric('best_epoch', best_epoch)

        probs = softmax(logits, axis=1)[:, 1]
        predictions = (probs > 0.5).astype(int)

        self.evaluation_data = {
            'evaluation_logits': logits.tolist(),
            'evaluation_probabilities': probs.tolist(),
            'evaluations_predictions': predictions.tolist(),
            'evaluation_labels': labels.tolist(),
            'evaluation_metrics': metrics}
        self.save_evaluation_data()
        return self.evaluation_data

    def save_evaluation_data(self):
        artifacts_path = utils.get_current_run_artifacts_path()
        if artifacts_path is None:
            self.logger.info('Could not save evaluation data, because artifacts path is None.')
            return False
        evaluation_artifacts_path = os.path.join(artifacts_path, 'evaluation')
        Path(evaluation_artifacts_path).mkdir(parents=True, exist_ok=True)
        evaluation_data_path = os.path.join(evaluation_artifacts_path, 'evaluation_data')
        with open(evaluation_data_path + '.pkl', 'wb') as f:
            self.logger.info(f'Saving evaluation data in pickle...')
            pickle.dump(self.evaluation_data, f)
        with open(evaluation_data_path + '.json', 'w', encoding='utf-8') as f:
            self.logger.info(f'Saving evaluation data in json...')
            json.dump(self.evaluation_data, f)

        for data_name, data in self.evaluation_data.items():
            evaluations_data_path_part = os.path.join(evaluation_artifacts_path, data_name)
            with open(evaluations_data_path_part + '.json', 'w', encoding='utf-8') as f:
                json.dump(data, f)
        self.logger.info('Successfully saved evaluation data.')
        return True
