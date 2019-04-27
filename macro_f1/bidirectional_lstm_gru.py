#! /usr/bin/env python

import numpy as np
from time import time
from gensim.models import Word2Vec
from keras import optimizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Bidirectional
from keras.layers import LSTM
from sklearn.model_selection import StratifiedKFold

from data_prep import \
    load_csv_data, \
    create_data_dictionary, \
    prepare_data_for_training_lstm
from metrics import Metrics

hps = {
    'lr': 0.001,
    'cv_num_splits': 5,
    'dropout': 0.5,
    'recurrent_dropout': 0.2,
    'epochs': 10,
    'embedding_size': 200,
    'pad_length': 200,
    'batch_size': 128
}


def main():
    print('Loading pretrained Word2Vec embeddings for wiki toxicity data...')
    embeddings = Word2Vec.load('../data/personal_attacks/output/word2vec_model.model')
    vocab = np.load('../data/personal_attacks/output/wikimedia-personal-attacks-embeddings.npy')
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
    results_file_name = f'results/bilstm/results_{time()}.csv'
    results_file = open(results_file_name, mode='a')

    # Build, train, and evaluate the LSTM model
    results_file.write(Metrics.get_csv_header('bi-lstm', 'macro') + '\n')
    for train, test in strat_kfold.split(x, y):
        lstm_metrics = Metrics()
        model = build_lstm_model(len(vocab), vocab)
        model.fit(x[train], y[train], batch_size=hps['batch_size'],
                  epochs=hps['epochs'], callbacks=[lstm_metrics],
                  validation_data=(x[test], y[test]))

        acc = lstm_metrics.val_accs[-1]
        prec = lstm_metrics.val_precisions[-1]
        recall = lstm_metrics.val_recalls[-1]
        f1 = lstm_metrics.val_f1s[-1]
        results_file.write(f'lstm, {acc}, {prec}, {recall}, {f1}' + '\n')

    # Build, train, and evaluate the GRU model
    results_file.write(Metrics.get_csv_header('bi-gru', 'macro') + '\n')
    for train, test in strat_kfold.split(x, y):
        gru_metrics = Metrics()
        model = build_gru_model(len(vocab), vocab)
        model.fit(x[train], y[train], batch_size=hps['batch_size'],
                  epochs=hps['epochs'], callbacks=[gru_metrics],
                  validation_data=(x[test], y[test]))

        acc = gru_metrics.val_accs[-1]
        prec = gru_metrics.val_precisions[-1]
        recall = gru_metrics.val_recalls[-1]
        f1 = gru_metrics.val_f1s[-1]
        results_file.write(f'gru, {acc}, {prec}, {recall}, {f1}' + '\n')


def build_lstm_model(vocab_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=hps['embedding_size'],
                        weights=[pretrained_weights], trainable=False))
    model.add(Bidirectional(LSTM(units=hps['pad_length'], dropout=hps['dropout'],
                                 recurrent_dropout=hps['recurrent_dropout'])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(hps['lr']), metrics=['accuracy'])
    return model


def build_gru_model(vocab_size, pretrained_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=hps['embedding_size'],
                        weights=[pretrained_weights], trainable=False))
    model.add(Bidirectional(GRU(units=hps['pad_length'], dropout=hps['dropout'],
                                recurrent_dropout=hps['recurrent_dropout'])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(hps['lr']), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
