from .model import Model

import mlflow
import numpy as np
from scipy.special import softmax
import datasets as hf_datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from transformers.integrations import MLflowCallback

class BertBasedUncased(Model):
    def __init__(self, random_state, logger):
        self.is_fit = False
        self.name = "bert-base-uncased"
        mlflow.log_param('model_name', self.name)
        super().__init__(random_state, logger)

    def fit(self, dataset):
        self.dataset = dataset
        set_seed(self.random_state)
        train, val, test = self.__prepare_dataset(dataset)
        tokenizer = BertTokenizer.from_pretrained(self.name)

        def tokenize_function(example):
            return tokenizer(example["article"], truncation=True, padding="max_length", max_length=256)

        self.train_tokenized = train.map(tokenize_function, batched=True)
        self.val_tokenized = val.map(tokenize_function, batched=True)
        self.test_tokenized = test.map(tokenize_function, batched=True)

        self.train_tokenized.set_format(type="torch")
        self.val_tokenized.set_format(type="torch")
        self.test_tokenized.set_format(type="torch")
        self.train(self.train_tokenized, self.val_tokenized)
        self.is_fit = True

    def train(self, train, val):
        self.model = BertForSequenceClassification.from_pretrained(self.name, num_labels=2)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            probs = softmax(logits, axis=1)[:, 1]
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions)
            recall = recall_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            roc_auc = roc_auc_score(labels, probs)
            cm = confusion_matrix(labels, predictions)
            tn, fp, fn, tp = cm.ravel()
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            return {"false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate,
                    "accuracy": accuracy, 'precision': precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}

        artifacts_dir = f"mlruns/{mlflow.active_run().info.experiment_id}/{mlflow.active_run().info.run_id}/artifacts/"
        self.training_args = TrainingArguments(
            output_dir=artifacts_dir + 'model/checkpoints',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            logging_steps=10,
            save_steps=100,
            weight_decay=0.01
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics=compute_metrics,
            callbacks=[MLflowCallback()]
        )
        self.trainer.train()
        self.trainer.save_model(artifacts_dir + 'model/')

    def evaluate(self):
        self.trainer.evaluate(self.test_tokenized, metric_key_prefix="test")

    def __prepare_dataset(self, dataset):
        train, val, test = dataset.split()
        return (hf_datasets.Dataset.from_pandas(train),
                hf_datasets.Dataset.from_pandas(val),
                hf_datasets.Dataset.from_pandas(test))
