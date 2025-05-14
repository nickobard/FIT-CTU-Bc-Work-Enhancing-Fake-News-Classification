import utils
from experiments.metrics import Metric, standard_evaluation_metrics
from ..base import TransformersModels
from scipy.special import softmax
import mlflow
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed, AutoConfig, \
    AutoModelForSequenceClassification
from ..callbacks import HF_CustomMLflowCallback
from ..trainer import TrainerWithEmbeddingsCollection
import os
import glob
import pickle
from ....experiments.metrics import compute_standard_metrics, Loss
from pathlib import Path
import json
from ....utils import log_params
from ordered_set import OrderedSet


class BertBaseUncased(TransformersModels):

    def __init__(self, training_arguments=None,
                 train_best_model_metric=Loss,
                 evaluation_metrics=standard_evaluation_metrics):
        super().__init__(train_best_model_metric)
        self.name = "bert-base-uncased"
        training_arguments = {} if training_arguments is None else training_arguments
        self.training_args = {**self.get_default_training_args(), **training_arguments}
        self.evaluation_metrics = evaluation_metrics
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
        self.logger.info(f'Model name: {self.name}')
        log_params({'model_input_hyperparameters': self.training_args}, self.logger)
        self.logger.info(f'Model input hyperparameters: {self.training_args}')
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
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=2)

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
            load_best_model_at_end=False,
            metric_for_best_model=f'eval_{self.train_best_model_metric.name}',
            greater_is_better=self.train_best_model_metric.greater_is_better
        )

        callbacks = [HF_CustomMLflowCallback()]
        early_stopping_patience = self.training_args['early_stopping_patience']
        if early_stopping_patience is not None:
            early_stopping_threshold = self.training_args['early_stopping_threshold']
            esc = EarlyStoppingCallback(early_stopping_patience,
                                        early_stopping_threshold)
            callbacks.append(esc)

        self.trainer = TrainerWithEmbeddingsCollection(
            model=self.model,
            args=training_args,
            train_dataset=dataset.preprocessed_train_set.dataset,
            eval_dataset=dataset.preprocessed_val_set.dataset,
            compute_metrics=compute_standard_metrics,
            callbacks=callbacks,
            logger=self.logger
        )
        self.trainer.train(resume_from_checkpoint=True if self.has_checkpoints() else False)

    def evaluate(self):
        for evaluation_metric in self.evaluation_metrics:
            self.logger.info(f'Evaluating model using {evaluation_metric.name} metric...')
            best_entry = self._load_best_model_for_metrics(OrderedSet([evaluation_metric, Loss]))
            self.logger.info(f'Best entry according to validation metrics: {best_entry}')
            self.logger.info(f'Best model found at epoch {best_entry["epoch"]}.')
            # Extract predictions and labels
            with self.trainer.collect_hidden_states():
                logits, labels, metrics = self.trainer.predict(self.dataset.preprocessed_test_set.dataset,
                                                               metric_key_prefix='test')
                embeddings = self.trainer.get_collected_embeddings()
            best_epoch = best_entry['epoch']
            metrics['test_epoch'] = best_epoch
            self.logger.info(f'Test metrics: {metrics}')
            postfix = f'_by_{evaluation_metric.name}'
            mlflow.log_metrics({m_key + postfix: m_val for m_key, m_val in metrics.items()})
            mlflow.log_metric('best_epoch' + postfix, best_epoch)

            probs = softmax(logits, axis=1)[:, 1]
            predictions = (probs > 0.5).astype(int)

            self.evaluation_data = {
                'evaluation_embeddings': embeddings.tolist(),
                'evaluation_logits': logits.tolist(),
                'evaluation_probabilities': probs.tolist(),
                'evaluations_predictions': predictions.tolist(),
                'evaluation_labels': labels.tolist(),
                'evaluation_metrics': metrics}
            self.save_evaluation_data(evaluation_metric)
        self.logger.info('Finished model evaluations stage.')
        return self

    def save_evaluation_data(self, metric):
        artifacts_path = utils.get_current_run_artifacts_path()
        if artifacts_path is None:
            self.logger.info('Could not save evaluation data, because artifacts path is None.')
            return False
        evaluation_artifacts_path = os.path.join(artifacts_path, 'evaluation', f'by_{metric.name}')
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

    def _load_best_model_for_metrics(self, metrics: OrderedSet[Metric]):
        # 1. Filter log entries that have all required metrics at evaluation-time
        metric_names = [f'eval_{metric.name}' for metric in metrics]
        entries = [
            entry for entry in self.trainer.state.log_history
            if all(metric_name in entry for metric_name in metric_names) and entry.get("epoch") is not None
        ]
        if not entries:
            raise ValueError(f"No entries found for metrics: {metric_names}")

        def _get_entry_sort_key(entry, metrics):
            return tuple(entry[f'eval_{m.name}'] if m.greater_is_better else -entry[f'eval_{m.name}']
                         for m in metrics)

        # 2. Find the best entry using lexicographical sorting based on provided metrics
        best_entry = max(entries, key=lambda e: _get_entry_sort_key(e, metrics))

        # 3. Derive checkpoint name from the global step
        step = int(best_entry["step"])
        ckpt_dir = os.path.join(self.trainer.args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
        self.logger.info(f"Found checkpoint with best metrics at: checkpoint-{step}")

        # 4. Load and return the model
        # 1) Load the exact config you used
        config = AutoConfig.from_pretrained(ckpt_dir)

        # 2) Instantiate the model from that config + weights
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir,
            config=config
        )
        # 3) Move to the right device & set to eval mode
        model.to(self.trainer.args.device)
        model.eval()

        # 4) Swap in and return
        self.model = model
        self.trainer.model = model
        return best_entry

    def _params(self):
        super_class_params = super()._params()
        additional_params = {'model_name': self.name,
                             'main_metrics_name': self.train_best_model_metric.name}
        return {**super_class_params,
                **additional_params,
                **self.training_args}
