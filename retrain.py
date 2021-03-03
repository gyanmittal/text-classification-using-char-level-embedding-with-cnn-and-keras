import numpy as np
import os
from util import load_data_and_labels_from_csv_file, pad_sentences, text_to_sequence, load_vocab_json, generate_char_level_features
import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
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

# Shuffle data
np.random.seed(1) #same shuffling each time
shuffle_indices = np.random.permutation(np.arange(len(labels)))
x = x[shuffle_indices]
labels = labels[shuffle_indices]

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
#checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=checkpoint_dir, histogram_freq=1)
adam = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
print("Traning Model...")
epochs = 5
batch_size = 32
validation_split = 0.1
verbose = 1

model.fit(x, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=[checkpoint, tensorboard_callback])

