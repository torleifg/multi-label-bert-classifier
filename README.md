# Multi-Label Genre and Form Classification of Book Reviews

This repository contains code and resources for fine-tuning a BERT-based model for multi-label genre and form
classification of book reviews. It uses BERT-based language models from Nasjonalbiblioteket (Norwegian National
Library), DistilBERT and a dataset drawn from the open API of Biblioteksentralen. The dataset is highly imbalanced.

## Overview

- **Multi-Label Classification**: Each book review can belong to multiple genre and form.
- **Fine-Tuning BERT**: The model is fine-tuned using a chosen BERT-based langauge model.
- **Evaluation**: The model is evaluated using metrics such as **F1 macro score**.

## Resources

- **Bibbi Metadata REST API**: Used for collecting book metadata, including reviews, genre and form
  labels (https://bibliografisk.bs.no/).
- **Norwegian Thesaurus on Genre and Form**: Used for the genre and form
  vocabulary (https://www.nb.no/nbvok/ntsf/en/).
- **NB-BERT-base**: Pre-trained Norwegian language model used for
  fine-tuning (https://huggingface.co/NbAiLab/nb-bert-base).
- **NB-BERT-large**: Pre-trained Norwegian language model used for
  fine-tuning (https://huggingface.co/NbAiLab/nb-bert-large).
- **DistilBERT base multilingual (cased)**: Pre-trained language model used for
  fine-tuning (https://huggingface.co/distilbert/distilbert-base-multilingual-cased).

## Getting Started

### 1. Install Python (Mac)

Install pyenv:

```bash
brew install pyenv
```

Install xz (if using M1 or M2 Mac):

```bash
brew install xz
```

Install Python (max version 3.12.*):

```bash
pyenv install 3.12.7     
```

Switch to Python version:

```bash
pyenv global 3.12.7     
```

Verify Python version
```bash
python --version  
```

### 2. Set Up the Virtual Environment

In the root folder of the project. Start by creating a virtual environment for managing dependencies:

```bash
python -m venv env
```

Activate the virtual environment:

```bash
source env/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 3. Install JupyterLab Desktop

https://github.com/jupyterlab/jupyterlab-desktop

Open the project in JupyterLAb and activate the newly created virtual environment (upper right corner). 

### 4. Create Dataset

The dataset contains metadata including reviews and associated genre and form labels. Since the dataset is highly
imbalanced, techniques such as oversampling, undersampling, or data augmentation may be applied to improve the
performance of the model.

Run the ```create_dataset.ipynb``` notebook to create the dataset.

### 5. Describe Dataset

Run the ```describe_dataset.ipynb``` notebook to explore and visualize the dataset distribution.

### 6. Fine-Tuning

Choose and fine-tune a model by running the ```fine_tune_model.ipynb``` notebook. This notebook will:

* load the dataset.
* process the data for multi-label classification.
* handle data imbalance using appropriate techniques.
* fine-tune the model on the prepared dataset.

### 7. Classification

Once the model has been fine-tuned, you can use the ```genre_classification.ipynb``` notebook to classify new book
reviews into genre and form. This notebook allows you to:

- load the fine-tuned model and checkpoint.
- input book reviews for genre classification.
- output the predicted genre and form labels for the reviews.

## 8. Evaluation and F1 Macro Score

The model performance is evaluated using several metrics, including F1 Macro Score, which is particularly suited for
imbalanced datasets like this one.

After training the NB-Bert-base model for one epoch, the F1 Macro Score was: **0.83**.

After training the NB-Bert-large model for one epoch, the F1 Macro Score was: **0.89**.

After training the DistilBERT base multilingual (cased) for one epoch, the F1 Macro Score was: **0.84**.
