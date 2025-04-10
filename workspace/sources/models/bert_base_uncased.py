from models.model import Model

import mlflow
import numpy as np
from scipy.special import softmax
import datasets as hf_datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from transformers.integrations import MLflowCallback
import utils
import os


class BertBasedUncased(Model):

    @classmethod
    def load_from_mlflow(cls, logger):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {cls.__name__} class.")

    @classmethod
    def mlflow_model_artifact_exists(cls, logger):
        local_path = cls.get_model_artifacts_path()
        model_path = os.path.join(local_path, 'model.pkl')
        if os.path.exists(model_path):
            return True
        else:
            logger.error(f"Error: Model artifact not found at path: {model_path}")
            return False

    def save_to_mlflow(self):
        raise NotImplementedError(
            f"The save_to_mlflow method is not supported for the {self.__class__.__name__} class.")

    def __init__(self, random_state, logger):
        self.is_fit = False
        self.name = "bert-base-uncased"
        mlflow.log_param('model_name', self.name)
        super().__init__(random_state, logger)

    def saved_datasets_exist(self):
        return all(os.path.exists(os.path.join(self.get_model_artifacts_path(), ds))
                   for ds in ["train_tokenized", "val_tokenized", "test_tokenized"])

    def load_saved_dataset(self, dataset_name):
        dataset_path = os.path.join(self.get_model_artifacts_path(), dataset_name)
        if os.path.exists(dataset_path):
            return hf_datasets.Dataset.load_from_disk(dataset_path)
        else:
            raise FileNotFoundError(f"The dataset '{dataset_name}' does not exist at path: {dataset_path}")

    def fit(self, dataset):
        self.dataset = dataset
        set_seed(self.random_state)
        train, val, test = self.__prepare_dataset(dataset)
        tokenizer = BertTokenizer.from_pretrained(self.name)

        def tokenize_function(example):
            return tokenizer(example["article"], truncation=True, padding="max_length", max_length=256)

        if self.saved_datasets_exist():
            self.train_tokenized = self.load_saved_dataset("train_tokenized")
            self.val_tokenized = self.load_saved_dataset("val_tokenized")
            self.test_tokenized = self.load_saved_dataset("test_tokenized")
        else:
            self.train_tokenized = train.map(tokenize_function, batched=True)
            self.val_tokenized = val.map(tokenize_function, batched=True)
            self.test_tokenized = test.map(tokenize_function, batched=True)
            self.train_tokenized.set_format(type="torch")
            self.val_tokenized.set_format(type="torch")
            self.test_tokenized.set_format(type="torch")
            # Save tokenized datasets
            self.train_tokenized.save_to_disk(os.path.join(self.get_model_artifacts_path(), "train_tokenized"))
            self.val_tokenized.save_to_disk(os.path.join(self.get_model_artifacts_path(), "val_tokenized"))
            self.test_tokenized.save_to_disk(os.path.join(self.get_model_artifacts_path(), "test_tokenized"))

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

        output_dir = os.path.join([self.get_model_artifacts_path(), 'checkpoints'])
        self.training_args = TrainingArguments(
            output_dir=output_dir,
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

    def evaluate(self):
        self.trainer.evaluate(self.test_tokenized, metric_key_prefix="test")

    def __prepare_dataset(self, dataset):
        train, val, test = dataset.split()
        train_hf = hf_datasets.Dataset.from_pandas(train)
        val_hf = hf_datasets.Dataset.from_pandas(val)
        test_hf = hf_datasets.Dataset.from_pandas(test)

        # Save original datasets
        train_hf.save_to_disk(os.path.join(self.get_model_artifacts_path(), "train_dataset"))
        val_hf.save_to_disk(os.path.join(self.get_model_artifacts_path(), "val_dataset"))
        test_hf.save_to_disk(os.path.join(self.get_model_artifacts_path(), "test_dataset"))

        return train_hf, val_hf, test_hf
