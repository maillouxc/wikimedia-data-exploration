#! /usr/bin/env python

# Wikimedia Toxicity Personal Attacks Data Prep.

import sys
import os
import pathlib
import logging
import numpy as np
import pandas as pd
import msgpack
import re
import csv
from gensim.models import Word2Vec


logging.basicConfig(level=logging.INFO)
data_dir = r''
min_num_votes = 6
data_path = pathlib.Path(data_dir)
output_dir = pathlib.Path('output')

# Get the data, create dataframes from the tab-separated files.
attacks_comments_path = data_path / 'attack_annotated_comments.tsv'
attacks_labels_path = data_path / 'attack_annotations.tsv'
attacks_comments_df = pd.read_csv(attacks_comments_path, sep='\t', header=0)
attacks_labels_df = pd.read_csv(attacks_labels_path, sep='\t', header=0)
logging.debug(attacks_comments_df.head())
logging.debug(attacks_labels_df.head())

# Merge data frames of comments and annotations on rev_id.
attacks_merged = pd.merge(attacks_comments_df, attacks_labels_df, on='rev_id')
attacks_merged_summed = attacks_merged.groupby('rev_id').sum()

# LABELS: Build set of rev_ids that contain personal attacks as labels.
attacks = attacks_merged_summed.loc[attacks_merged_summed['attack'] > 5].copy()
attacks.reset_index(level=0, inplace=True)
attacks['attack'] = 1
attacks.drop(
    ['year', 'logged_in', 'worker_id', 'quoting_attack', 'recipient_attack',
     'third_party_attack', 'other_attack'], axis=1, inplace=True)

# Build set of rev_ids that do not contain attacks.
no_attacks = attacks_merged_summed.loc[attacks_merged_summed['attack'] <= 5].copy()
no_attacks.reset_index(level=0, inplace=True)
no_attacks['attack'] = 0
no_attacks.drop(
    ['year', 'logged_in', 'worker_id', 'quoting_attack', 'recipient_attack',
     'third_party_attack', 'other_attack'], axis=1, inplace=True)

# Combine the the two sets and sort.
labels = attacks.append(no_attacks)
labels.sort_values(by=['rev_id'], inplace=True)
labels.reset_index(level=0, drop=True, inplace=True)

logging.debug(print(labels.head()))

# FEATURES: Create features.
features = attacks_merged.groupby('rev_id').first().copy()

# Reset index, saving rev_id as column.
features.reset_index(level=0, inplace=True)

# Drop everything except for 'rev_id' and 'comment'.
features.drop(['year', 'logged_in', 'ns', 'sample', 'split', 'worker_id',
               'quoting_attack', 'recipient_attack', 'third_party_attack',
               'other_attack', 'attack'], axis=1, inplace=True)

# Merge with labels for complete set labeled data.
features = pd.merge(features, labels, on='rev_id').copy()

# Write features and labels to disk.
csv_path = pathlib.Path('raw_features.csv')
features.to_csv(csv_path)

logging.debug(features.head())

# Build vocabulary and word embeddings from source.

# Store records
labels = []
tokens = []

# Process csv one line at a time
with open(csv_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    lineno = 0
    idx = 0
    for line in csv_reader:
        # Skip header.
        lineno += 1
        logging.debug(print("Processing line %i     \r" % lineno))

        # Begin at index 1.
        idx += 1

        text = line['comment']
        text = text.lower()
        text = re.sub("NEWLINE_TOKEN", '', text)
        text = re.sub("TAB_TOKEN", '', text)
        text = re.sub("`", '', text)
        text = re.sub("'", '', text)
        text = re.sub("\.{2,}", '.', text)
        text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
        text = re.sub('\.', ' . ', text)
        text = re.sub('\?', ' ? ', text)
        text = re.sub('!', ' ! ', text)
        text = text.split()

        # Drop empty comments.
        if len(text) == 0:
            continue

        # Split into sentences.
        sentences = []
        sentence = []
        for t in text:
            # Use '.', '!', '?' as markers of end of sentence.
            if t not in ['.', '!', '?']:
                sentence.append(t)
            else:
                sentence.append(t)
                sentences.append(sentence)
                sentence = []

        # If sentence has word, add to list of sentences.
        if len(sentence) > 0:
            sentences.append(sentence)

        # Add split sentences to tokens.
        tokens.append(sentences)

        # Add label.
        labels.append(line['attack'])

# Use all processed raw text to train word2vec.
# TODO Incorporate FastText or other algos, too.
allsents = [sent for doc in tokens for sent in doc]
embedding_size = 200
model = Word2Vec(allsents, min_count=5, size=embedding_size, workers=4, iter=5)
model.init_sims(replace=True)

# Save all word embeddings to matrix
vocab = np.zeros((len(model.wv.vocab) + 1, embedding_size))
word2id = {}

# First row of embedding matrix isn't used so that 0 can be masked.
for key, val in model.wv.vocab.items():
    # Begin indexes with offset of 1.
    idx = val.__dict__['index'] + 1

    # Build 2D np array (idx, vector)
    vocab[idx, :] = model[key]

    # Dictionary mapping word to index.
    word2id[key] = idx

# Normalize embeddings.
vocab -= vocab.mean()
vocab /= (vocab.std() * 2)
# Reset first row to 0
vocab[0, :] = np.zeros(embedding_size)

# Add additional word embedding for unknown words.

unk = len(vocab) - 1

# Convert words to word indices.
data = {}
for idx, doc in enumerate(tokens):
    sys.stdout.write('processing %i of %i records       \r' % (idx + 1, len(tokens)))
    sys.stdout.flush()
    dic = {'label': labels[idx], 'text': doc}

    # Build list of indices representing the words of each sentence,
    # if word is a key in word2id mapping, use unk, defined: vocab[len(vocab)-1].
    indices = []
    for sent in doc:
        indices.append([word2id[word] if word in word2id else unk for word in sent])

    # Add indices to dictionary.
    dic['idx'] = indices

    # Add dictionary containing label, text, indices to data dictionary at index.
    data[idx] = dic

# Create output dir if it doesn't exist.
try:
    os.makedirs(output_dir)
except FileExistsError:
    logging.info('Output directory already exists.')

# Write data dictionary to file.
data_path = output_dir / 'wikimedia-personal-attacks-data.bin'
with open(data_path, 'wb') as f:
    msgpack.pack(data, f)

# Write embeddings to file in numpy binary format.
embeddings_path = output_dir / 'wikimedia-personal-attacks-embeddings'
np.save(embeddings_path, vocab)

print('Loading preprocessed training/test data...')
data_path = 'data/personal_attacks/output/wikimedia-personal-attacks-data.bin'
with open(data_path, 'rb') as data_file:
    unpacked_data = msgpack.unpack(data_file.read())
print(unpacked_data)
