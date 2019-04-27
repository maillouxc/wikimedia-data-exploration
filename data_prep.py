import csv
import re
import numpy as np


def prepare_data_for_training_lstm(data):
    x = []
    y = []
    for value in data.values():
        text_indexes = []
        for item in value['idx']:
            for subitem in item:
                text_indexes.append(subitem)
        label = int(value['label'])
        x.append(text_indexes)
        y.append(label)
    return np.array(x), np.array(y)


def create_data_dictionary(tokens, labels, word2id, unk):
    print('Creating data dictionary...')
    data = {}
    for idx, doc in enumerate(tokens):
        dic = {'label': labels[idx], 'text': doc}

        # Build list of indices representing the words of each sentence
        indices = []
        for sent in doc:
            indices.append([word2id[word] if word in word2id else unk for word in sent])
        # Add indices to dictionary
        dic['idx'] = indices
        # Add dictionary containing label, text, indices to data dictionary at index
        data[idx] = dic
    return data


def load_csv_data():
    print('Loading CSV file containing labeled data...')
    data_path = '../data/personal_attacks/features.csv'
    with open(data_path, mode='r') as csv_file:
        labels = []
        tokens = []

        csv_reader = csv.DictReader(csv_file)
        lineno = 0
        idx = 0
        for line in csv_reader:
            lineno += 1
            idx += 1
            text = line['comment']
            text = text.lower()
            text = re.sub("NEWLINE_TOKEN", '', text)
            text = re.sub('newline_token', '', text)
            text = re.sub("TAB_TOKEN", '', text)
            text = re.sub("`", '', text)
            text = re.sub("'", '', text)
            text = re.sub("\.{2,}", '.', text)
            text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
            text = re.sub('\.', ' . ', text)
            text = re.sub('\?', ' ? ', text)
            text = re.sub('!', ' ! ', text)
            text = text.split()

            if len(text) == 0:
                continue

            # Split into sentences
            sentences = []
            sentence = []
            for t in text:
                if t not in ['.', '!', '?']:
                    sentence.append(t)
                else:
                    sentence.append(t)
                    sentences.append(sentence)
                    sentence = []

            if len(sentence) > 0:
                sentences.append(sentence)
            tokens.append(sentences)
            labels.append(line['attack'])

        return tokens, labels
