from abc import ABC
import mlflow
import numpy as np
from scipy.special import softmax
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from transformers.integrations import MLflowCallback


class Model(ABC):
    def __init__(self, random_state):
        self.random_state = random_state


class BertBasedUncased(Model):
    def __init__(self, random_state):
        self.name = "bert-base-uncased"
        mlflow.log_param('model_name', self.name)
        super().__init__(random_state)

    def fit(self, dataset):
        train, val, test = self.__prepare_dataset(dataset)
        tokenizer = BertTokenizer.from_pretrained(self.name)

        def tokenize_function(example):
            return tokenizer(example["article"], truncation=True, padding="max_length", max_length=256)

        train_tokenized = train.map(tokenize_function, batched=True)
        val_tokenized = val.map(tokenize_function, batched=True)
        test_tokenized = test.map(tokenize_function, batched=True)

        train_tokenized.set_format(type="torch")
        val_tokenized.set_format(type="torch")
        test_tokenized.set_format(type="torch")

    def train(self, train, val):
        model = BertForSequenceClassification.from_pretrained(self.name, num_labels=2)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            probs = softmax(logits, axis=1)[:, 1]
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions)
            recall = recall_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            roc_auc = roc_auc_score(labels, probs)
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            return {"false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate, "accuracy": accuracy, 'precision': precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}

        training_args = TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            logging_steps=10,
            save_steps=100,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics=compute_metrics,
            callbacks=[MLflowCallback()]
        )

    def __prepare_dataset(self, dataset):
        train, val, test = dataset.split()
        return Dataset.from_pandas(train), Dataset.from_pandas(val), Dataset.from_pandas(test)
