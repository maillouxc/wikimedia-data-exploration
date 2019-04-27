#! /usr/bin/env python

import csv
import os
import re

import fasttext
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate


def main():
    estimator = FasttextBinaryClassifier(model_dir="../data/personal_attacks/fasttext")
    texts, labels = load_features()

    # FastText takes labels as strings, but sklearn expects ints - we have to correct for this
    accuracy_scorer = make_scorer(accuracy_score)
    precision_scorer = make_scorer(precision_score, pos_label="1")
    recall_scorer = make_scorer(recall_score, pos_label="1")
    f1_scorer = make_scorer(f1_score, average='macro', pos_label="1")

    scoring = {
        'accuracy': accuracy_scorer,
        'precision': precision_scorer,
        'recall': recall_scorer,
        'f1': f1_scorer
    }
    scores = cross_validate(estimator, texts, labels,
                            cv=StratifiedKFold(n_splits=5),
                            scoring=scoring, return_train_score=False)

    print('Run title, Accuracy, Precision, Recall, f1')
    for i in range(0, 5):
        run_title = 'fasttext'
        acc = scores['test_accuracy'][i]
        prec = scores['test_precision'][i]
        recall = scores['test_recall'][i]
        f1 = scores['test_f1'][i]
        print(f'{run_title}, {acc}, {prec}, {recall}, {f1}')


def load_features():
    print('Loading CSV file containing labeled data...')
    data_path = '../data/personal_attacks/features.csv'
    labels = []
    raw_text_strings = []
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        lineno = 0
        idx = 0
        for line in csv_reader:
            lineno += 1  # Skip header
            idx += 1  # Begin at index 1

            text = line['comment']
            text = text.lower()
            text = re.sub("NEWLINE_TOKEN", '', text)
            text = re.sub("newline_token", '', text)
            text = re.sub("TAB_TOKEN", '', text)
            text = re.sub("`", '', text)
            text = re.sub("'", '', text)
            text = re.sub("\.{2,}", '.', text)
            text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
            text = re.sub('\.', ' . ', text)
            text = re.sub('\?', ' ? ', text)
            text = re.sub('!', ' ! ', text)
            line_tokens = text.split()

            # Drop empty comments
            if len(line_tokens) <= 3:
                continue

            labels.append(line['attack'])
            raw_text_strings.append(str.join(' ', line_tokens))

    return raw_text_strings, labels


def to_fasttext_label_format(label):
    """Takes a string and returns it in the format expected by FastText for it's labels."""
    return f'__label__{label}'


def from_fasttext_label_format(label):
    """Takes a string and removes the FastText label prefix"""
    label_prefix = '__label__'
    return label[label.startswith(label_prefix) and len(label_prefix):]


def store_data_in_fasttext_file_format(output_file_name, features, labels):
    """
    Write the training data in the Fasttext format to disk.
    :param output_file_name: The name of the file to write the training data to.
    :param features: The features, a list of strings.
    :param labels: The labels associated with the features.
    """
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for i in range(0, len(features)):
            f.write(f"{to_fasttext_label_format(labels[i])} {features[i]}\n")


class FasttextBinaryClassifier(BaseEstimator):

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None

    def fit(self, features, labels):
        """Trains the fasttext classifier on the provided data and outputs the results."""
        store_data_in_fasttext_file_format(
            os.path.join(self.model_dir, "train.txt"), features, labels)
        fasttext.supervised(os.path.join(self.model_dir, "train.txt"),
                            os.path.join(self.model_dir, "cv_model"),
                            label_prefix='__label__', bucket=2000000,
                            epoch=10, dim=300, lr=0.005)
        self.model = fasttext.load_model(os.path.join(self.model_dir, 'cv_model.bin'))
        return self

    def predict(self, sample):
        predictions = self.model.predict(sample)
        formatted_predictions = []
        for prediction in predictions:
            formatted_predictions.append(from_fasttext_label_format(prediction[0]))
        return formatted_predictions


if __name__ == '__main__':
    main()
