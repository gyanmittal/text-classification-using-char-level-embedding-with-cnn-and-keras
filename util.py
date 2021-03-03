import numpy as np
import re
import itertools
from collections import Counter
import json
import math
import random

def process_text(text):
    text = text.strip().lower()
    text = text.replace("<br />", " ")
    text = text.replace("<br/>", " ")
    text = ' ' + text + ' '
    text = re.sub(r"[^A-Za-z0-9\'\.*$@#]", " ", text) #replace all the characters with space except mentioned here
    text = re.sub(r"['.]", "", text) # Remove . and '
    text = re.sub(r" +", " ", text) #replace muliple space with single space
    text = re.sub(r' [0-9 ]+ \s*', ' <n/> ', text)  # Replace with special notation in case only digits
    text = re.sub(r' [^a-z]+ \s*', ' <sd/> ', text)    # Replace with special notation in case only special character, with or without digit
    text = re.sub(r'(.)\1+', r'\1\1', text) #strip in case of consecutive more than 2 characters to two characters
    text = re.sub(r'([^a-z])\1+', r'\1', text) #strip in case of special characters occur more than once
    return text

def load_data_and_labels_from_csv_file(csv_file):
    lines = list(open(csv_file, 'r', encoding='utf-8').readlines())
    lines = [s.strip() for s in lines]
    labels, sentences = zip(*[line.split("\t") for line in lines])
    labels = np.array([1 if label=='spam' else 0 for label in labels ])
    return labels, np.array(sentences)

def generate_char_level_features(sentences, max_chars_features=300):
    lines_chars_level_features = [list(sentence.lower()) for sentence in sentences]
    lines_truncated_chars_level_features = [line_features if(len(line_features) <= max_chars_features) else line_features[:max_chars_features] for line_features in lines_chars_level_features]
    return lines_truncated_chars_level_features

def pad_sentences(sentences, padding_word='<PAD/>', max_sequence_length=200, is_max_sequence_length_modifiable=True):
   
    padded_sentences = []
    max_length = max([len(sentence) for sentence in sentences])
    if (max_length < max_sequence_length and is_max_sequence_length_modifiable==True): max_sequence_length = max_length 
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max_sequence_length - len(sentence)
        if(num_padding > 0):
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, max_vocab_size=20000, min_word_freq=1, padding_word='<PAD/>', unknown_word='<UNK/>'):
    word_counts = Counter(itertools.chain(*sentences)) # Count words
    vocabulay_inv = [[unknown_word, math.inf], [padding_word,math.inf]] + [[x[0], x[1]] for x in word_counts.most_common()] # Sort the word as frequency order
    if(len(vocabulay_inv) > (max_vocab_size+2)):
        vocabulay_inv = vocabulay_inv[:(max_vocab_size+2)]
    vocabulary = {word[0]: [i,word[1]] for i, word in enumerate(vocabulay_inv) if word[1] >= min_word_freq} # Build vocabulary, word: index
    return vocabulary

def text_to_sequence(sentences, vocabulary, unknown_word='<UNK/>'):
    x = np.array([[vocabulary[word][0] if word in vocabulary else vocabulary[unknown_word][0] for word in sen] for sen in sentences])
    return x

def save_vocab_json(file_path, word2id, params):
    json.dump(dict(src_word2id=word2id, params=params), open(file_path, 'w'), indent=2)

def load_vocab_json(file_path):
    entry = json.load(open(file_path, 'r'))
    src_word2id = entry['src_word2id']
    params = entry['params']
    return src_word2id, params

def precision_recall_f1_score(y_true, y_pred):
    y_pred = [1 if prediction >= .5 else 0 for prediction in y_pred]
    total_elem = len(y_true)
    true_positive = 0
    false_positive = 0
    total_true_positives = 0
    count = 0
    for y_true_elem in (y_pred):
        if (y_pred[count] == 1):
            if (y_true[count] == 1):
                true_positive += 1
            else:
                false_positive += 1
        total_true_positives += y_true[count]
        count += 1

    precision = 0
    recall = 0
    f1_score = 0
    if(true_positive+false_positive > 0):
        precision = true_positive/(true_positive+false_positive) * 100
    if (total_true_positives > 0):
        recall = true_positive * 100/ total_true_positives
    if (precision > 0 or recall > 0):
        f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score
