# Enhancing Fake News Classification with Advanced NLP Models

This repository contains the source code, experiments, materials, and thesis sources for the bachelor's thesis **"Enhancing Fake News Classification with Advanced NLP Models"**.

The thesis extends previous research on fake news classification by replacing classical machine learning approaches with modern transformer-based architectures. The work evaluates models such as **BERT**, **DistilBERT**, **RoBERTa**, and **GPT-2** on multiple fake news datasets and compares their performance against previously published results.

In addition to model evaluation, the project provides a reusable experimental framework for training, preprocessing, experiment tracking, and result analysis using **MLflow**.

The final thesis document is available in:

```
fake_news_detection.pdf
```

## Repository Structure

### Thesis

```
fake_news_detection.pdf
```

Final version of the bachelor's thesis.

### LaTeX Sources

```
latex/
├── ctufit-thesis.tex
├── ctufit-thesis.cls
├── text/
├── utils/
└── ...
```

Contains the complete LaTeX source code used to generate the final thesis document.

### Reference Materials

```
materials/
├── BP Flajžík Jan/
└── Texty HowTo/
```

Supporting materials used during development of the thesis, including the previous bachelor's thesis by Jan Flajžík, on which this work is based.

### Experimental Workspace

```
workspace/
├── mlruns/
├── notebooks/
├── requirements.txt
├── sources/
└── tests/
```

Main directory containing the implementation and experimental environment.

#### MLflow Experiments

```
workspace/mlruns/
```

MLflow experiment tracking data, including logged runs, metrics, parameters, and artifacts.

#### Notebooks

```
workspace/notebooks/
```

Jupyter notebooks used for:

* experiment execution,
* exploratory analysis,
* visualization,
* hyperparameter tuning,
* result inspection.

#### Source Code

```
workspace/sources/
```

Reusable Python modules implementing:

* dataset loading,
* preprocessing pipelines,
* tokenization,
* transformer model integration,
* training logic,
* evaluation utilities.

#### Tests

```
workspace/tests/
```

Tests and validation scripts for selected project components.

#### Dependencies

```
workspace/requirements.txt
```

Python package requirements needed to run the experiments.

## Thesis Summary

Fake news detection has become an important research area due to the rapid spread of misinformation through online platforms. This work investigates whether modern transformer-based architectures can improve fake news classification performance compared to traditional machine learning methods.

The experiments are conducted on the same datasets used in previous research, enabling direct comparison between classical approaches and transformer models. The evaluated architectures include:

* BERT
* DistilBERT
* RoBERTa
* GPT-2

The project also introduces a modular experimentation framework with integrated MLflow support for experiment tracking, reproducibility, and result analysis.

## Author

**Nikita Bardatskii**

Bachelor's Thesis
Faculty of Information Technology
Czech Technical University in Prague
