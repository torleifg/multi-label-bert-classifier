# Multi-Label Genre and Form Classification of Book Reviews

This repository contains code and resources for fine-tuning a BERT-based model for multi-label genre and form
classification of book reviews. It uses the NB-BERT-large language model from Nasjonalbiblioteket (Norwegian National
Library) and a dataset drawn from the open API of Biblioteksentralen. The dataset is highly imbalanced.

## Overview

- **Multi-Label Classification**: Each book review can belong to multiple genre and form.
- **Fine-Tuning BERT**: The model is fine-tuned using NB-BERT-large.
- **Evaluation**: The model is evaluated using metrics such as **F1 macro score**.

## Resources

- **Bibbi Metadata REST API**: Used for collecting book metadata, including reviews, genre and form
  labels (https://bibliografisk.bs.no/).
- **Norwegian Thesaurus on Genre and Form**: Used for the genre and form
  vocabulary (https://www.nb.no/nbvok/ntsf/en/).
- **NB-BERT-large**: Pre-trained Norwegian language model used for
  fine-tuning (https://huggingface.co/NbAiLab/nb-bert-large).

## Getting Started

### 1. Set Up the Virtual Environment

Start by creating a virtual environment for managing dependencies:

```bash
python3 -m venv env
```

Activate the virtual environment:

```bash
source env/bin/activate
```

### 2. Install Requirements

After activating the virtual environment, install the necessary dependencies:

```bash
pip3 install -r requirements.txt
```

### 3. Dataset

The dataset contains metadata including reviews and associated genre and form labels. Since the dataset is highly
imbalanced, techniques such as oversampling, undersampling, or data augmentation may be applied to improve the
performance of the model.

Run the ```describe_dataset.ipynb``` notebook to explore and visualize the dataset distribution.

### 4. Fine-Tuning

Fine-tune the NB-BERT-large model by running the ```fine_tune_model.ipynb``` notebook. This notebook will:

* load the dataset.
* process the data for multi-label classification.
* handle data imbalance using appropriate techniques.
* fine-tune the NB-BERT-large model on the prepared dataset.

### 5. Classification

Once the model has been fine-tuned, you can use the ```genre_classification.ipynb``` notebook to classify new book
reviews into genre and form. This notebook allows you to:

- load the fine-tuned model and checkpoint.
- input book reviews for genre classification.
- output the predicted genre and form labels for the reviews.

## Evaluation and F1 Macro Score

The model performance is evaluated using several metrics, including F1 Macro Score, which is particularly suited for
imbalanced datasets like this one.

After training the model for one epoch, the F1 Macro Score was: **0.89**.
