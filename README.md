# Multi-Label Genre Classification of Book Reviews

Fine-Tuning BERT (language model) for Multi-Label Classification of Genre. The dataset is highly imbalanced. Consider
oversampling, undersampling or data augmentation to improve the distribution of labels. The dataset is collected from
the open API of Biblioteksentralen (https://www.bibsent.no/). The base model is from
Nasjonalbiblioteket (https://www.nb.no/).

## Python (macOS)

```shell
brew install python
```

## Virtual environment

### Create virtual environment

```shell
python3 -m venv env
```

### Activate virtual environment

```shell
source env/bin/activate
```

## Requirements

```shell
pip3 install -r requirements.txt
```

## Fine-Tuning

Run the ```fine_tune_model``` notebook.

## Classification

Input the model and checkpoint from Fine-Tuning and runt the ```genre_classification``` notebook.