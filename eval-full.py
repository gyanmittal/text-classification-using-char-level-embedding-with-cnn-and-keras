import numpy as np
import keras
from util import load_vocab_json, load_vocab_json, load_data_and_labels_from_csv_file, pad_sentences, text_to_sequence, generate_char_level_features, precision_recall_f1_score
from keras import backend as K
import tensorflow as tf

#load model
model = keras.models.load_model("model/cp.ckpt/")
#print(model.summary())

vocab_file = "model/vocab.json"
vocabulary, params = load_vocab_json(vocab_file)
seq_len = params['max_chars_features'] 

data_file = "data/SMSSpamCollection"
labels, sentences = load_data_and_labels_from_csv_file(data_file)

lines_chars_level_features = generate_char_level_features(sentences, params['max_chars_features'])
lines_chars_level_features = np.array(lines_chars_level_features)

x = pad_sentences(lines_chars_level_features, max_sequence_length=seq_len, is_max_sequence_length_modifiable=False)
x = text_to_sequence(x, vocabulary)

print("Generate predictions")
predictions = model.predict(x)

print("Ham caught as Spam:")
count = 0
count_ham_as_spam = 0
for sentence in sentences:
    if (predictions[count] >= 0.5 and labels[count] == 0):
        print((count+1), "\t\t", sentence, "\t\t", predictions[count])
        count_ham_as_spam += 1
    count += 1 

if(count_ham_as_spam == 0): 
    print("None\n")

print("Spam classified as Ham:")
count = 0
count_spam_classified_as_ham = 0
for sentence in sentences:
    if (predictions[count] < 0.5 and labels[count] == 1):
        print((count+1), "\t\t", sentence, "\t\t", predictions[count])
        count_spam_classified_as_ham += 1
    count += 1
if(count_spam_classified_as_ham == 0): 
    print("None\n")

precision, recall, f1_score = precision_recall_f1_score(labels, predictions)
print("precision:\t", precision)
print("recall:\t", recall)
print("f1_score:\t", f1_score)

