import random
from collections import Counter

import numpy as np


def compute_label_frequencies(dataset, label_col="labels"):
    label_counts = Counter()

    for labels in dataset[label_col]:
        label_counts.update(labels)

    return label_counts


def compute_sample_weights(dataset, label_counts, label_col="labels"):
    def weight(labels):
        return max(1.0 / label_counts[label] for label in labels)

    weights = [weight(example) for example in dataset[label_col]]

    return np.array(weights) / np.sum(weights)


def oversample_dataset(dataset, weights, factor=1.5, seed=42):
    random.seed(seed)
    sample_size = int(len(dataset) * factor)
    indices = random.choices(range(len(dataset)), weights=weights, k=sample_size)

    return dataset.select(indices)
