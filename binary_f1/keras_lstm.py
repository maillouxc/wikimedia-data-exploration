#! /usr/bin/env python

import numpy as np
from time import time
from gensim.models import Word2Vec
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU
from keras.layers import LSTM
from sklearn.model_selection import StratifiedKFold
import keras_metrics as km

from wikimedia_data_exploration.data_prep import \
    load_csv_data, \
    create_data_dictionary, \
    prepare_data_for_training_lstm
from wikimedia_data_exploration.metrics import Metrics

hps = {
    'lr': 0.001,
    'cv_num_splits': 2,
    'dropout': 0.5,
    'recurrent_dropout': 0.2,
    'epochs': 3,
    'embedding_size': 200,
    'pad_length': 30,
    'batch_size': 128
}


def main():
    print('Loading pretrained Word2Vec embeddings for wiki toxicity data...')
    embeddings = Word2Vec.load('data/personal_attacks/output/word2vec_model.model')
    vocab = np.load('data/personal_attacks/output/wikimedia-personal-attacks-embeddings.npy')
    print(f'Vocab size = {len(vocab)}')

    # Build word2id lookup for vocabulary
    word2id = {}
    for key, val in embeddings.wv.vocab.items():
        idx = val.__dict__['index']
        vocab[idx, :] = embeddings[key]
        word2id[key] = idx

    # Load data to train the model with
    tokens, labels = load_csv_data()
    data = create_data_dictionary(tokens, labels, word2id, unk=len(vocab) - 1)
    x, y = prepare_data_for_training_lstm(data)

    # Pad/trim the input sequences to all be the same length
    print('Padding sequences (samples x time)...')
    x = sequence.pad_sequences(x, maxlen=hps['pad_length'])
    print('x shape:', x.shape)

    # Split the data for cross validation
    strat_kfold = StratifiedKFold(n_splits=hps['cv_num_splits'], shuffle=True)

    # Afterwards, we will print the results to a CSV file to be analyzed later
    results_file_name = f'results/results_{time()}.csv'
    results_file = open(results_file_name, mode='a')

    # Build, train, and evaluate the LSTM model
    results_file.write(Metrics.get_csv_header('lstm', 'binary-f1') + '\n')
    lstm_csv_logger = CSVLogger(f'results/lstm_history.csv', append=True, separator=';')
    lstm_cv_results = []
    for train, test in strat_kfold.split(x, y):
        model = build_lstm_model(len(vocab), vocab)
        model.fit(x[train], y[train], hps['batch_size'], epochs=hps['epochs'],
                  callbacks=[lstm_csv_logger])
        _, accuracy, precision, recall, f1 = model.evaluate(x[test], y[test], hps['batch_size'])
        result = Result('lstm', accuracy, precision, recall, f1)
        lstm_cv_results.append(result)
    output_results_to_csv_file(results_file, lstm_cv_results)

    # Build, train, and evaluate the GRU model
    results_file.write(Metrics.get_csv_header('gru', 'binary-f1') + '\n')
    gru_csv_logger = CSVLogger(f'results/gru_history.csv', append=True, separator=';')
    gru_cv_results = []
    for train, test in strat_kfold.split(x, y):
        model = build_gru_model(len(vocab), vocab)
        model.fit(x[train], y[train], hps['batch_size'], epochs=hps['epochs'],
                  callbacks=[gru_csv_logger])
        _, accuracy, precision, recall, f1 = model.evaluate(x[test], y[test], hps['batch_size'])
        result = Result('gru', accuracy, precision, recall, f1)
        gru_cv_results.append(result)
    output_results_to_csv_file(results_file, gru_cv_results)


def build_lstm_model(vocab_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=hps['embedding_size'],
                        weights=[pretrained_weights], trainable=False))
    model.add(LSTM(units=hps['pad_length'], dropout=hps['dropout'],
                   recurrent_dropout=hps['recurrent_dropout']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(hps['lr']),
                  metrics=['accuracy', km.binary_precision(), km.binary_recall(), km.binary_f1_score()])
    return model


def build_gru_model(vocab_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=hps['embedding_size'],
                        weights=[pretrained_weights], trainable=False))
    model.add(GRU(units=hps['pad_length'], dropout=hps['dropout'],
                  recurrent_dropout=hps['recurrent_dropout']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(hps['lr']),
                  metrics=['accuracy', km.binary_precision(), km.binary_recall(), km.binary_f1_score()])
    return model


class Result:

    def __init__(self, run_title, accuracy, precision, recall, f1):
        self.run_title = run_title
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def get_csv_line(self):
        column_values = [
            self.run_title,
            self.accuracy,
            self.precision,
            self.recall,
            self.f1
        ]
        return ','.join(map(str, column_values))


# noinspection PyListCreation
def output_results_to_csv_file(results_file, cv_results):
    for result in cv_results:
        results_file.write(f'{result.get_csv_line()}\n')


if __name__ == '__main__':
    main()
