# Multi-Label Genre Classification of Book Reviews

Fine-Tuning BERT (language model) for Multi-Label Classification of Genre. The dataset is highly imbalanced. Consider
oversampling, undersampling or data augmentation to improve the distribution of labels.

The dataset is collected from the open API of Biblioteksentralen (https://www.bibsent.no/). The BERT-model and the Genre
vocabulary is from Nasjonalbiblioteket (https://www.nb.no/).

**Resources**

* Bibbi Metadata REST API (https://bibliografisk.bs.no/)
* Norwegian thesaurus on genre and form (https://www.nb.no/nbvok/ntsf/en/)
* NB-BERT-large (https://huggingface.co/NbAiLab/nb-bert-large/)

## Python

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

## Dataset

Run the ```describe_dataset``` notebook for a visualization of the dataset.

## Fine-Tuning

Run the ```fine_tune_model``` notebook to Fine-Tune the model.

## Classification

Input the model and checkpoint from Fine-Tuning and run the ```genre_classification``` notebook.