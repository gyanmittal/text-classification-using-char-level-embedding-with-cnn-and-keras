import numpy as np
import os
from util import load_data_and_labels_from_csv_file, build_vocab, pad_sentences, text_to_sequence, save_vocab_json, generate_char_level_features
import keras
from keras.layers import Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam
from keras.models import Input, Model
import requests # This library is used to make requests to internet
import zipfile

data_file = "data/SMSSpamCollection"

#Download and unzip the data file in data directory in case it doesn't exists already
if not os.path.exists(data_file):
    data_file_dir = os.path.dirname(data_file)
    if not os.path.exists(data_file_dir): os.makedirs(data_file_dir)

    # We are storing url of dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(url, allow_redirects=True)
    zip_file_download = data_file_dir + '/smsspamcollection.zip'

    # We are writing the content of above request to 'iris.data' file
    open(zip_file_download, 'wb').write(r.content)
    
    #Extract the zip file
    with zipfile.ZipFile(zip_file_download,"r") as zip_ref:
        zip_ref.extractall(data_file_dir)

# Load data
print("Loading data...")
labels, sentences = load_data_and_labels_from_csv_file(data_file)

params = {'max_chars_features': 500} 

lines_chars_level_features = generate_char_level_features(sentences, params['max_chars_features'])
params['max_chars_features'] = max([len(lines) for lines in lines_chars_level_features])

lines_chars_level_features = np.array(lines_chars_level_features)

# Build vocabulary
print("Build the vocabulary")
vocabulary = build_vocab(lines_chars_level_features, max_vocab_size=10000)
#print(vocabulary)

# Pad sentence
print("Padding sentences...")
x_text = pad_sentences(lines_chars_level_features, max_sequence_length=params['max_chars_features'])

seq_len = len(x_text[0]) 
print("The sequence length is: ", seq_len)

# Represent sentence with char index, using char index to represent a sentence
x = text_to_sequence(x_text, vocabulary)

# Shuffle data
np.random.seed(1) #same shuffling each time
shuffle_indices = np.random.permutation(np.arange(len(labels)))
x = x[shuffle_indices]
labels = labels[shuffle_indices]

"""
## Build CNN model
"""
vocab_size_or_total_features = len(vocabulary) 

embed_dim = 300 
filter_sizes = [3,4,5,6,7]
num_filters = 512
drop_out = 0.5

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(seq_len,), dtype='int32')
embedding = Embedding(input_dim=vocab_size_or_total_features, output_dim=embed_dim, input_length=seq_len)(inputs)
reshape = Reshape((seq_len,embed_dim,1))(embedding)

# Defining Conv and Maxpool Layer for each filter_size
maxpool_for_each_filter = []
for filter_size in filter_sizes:
    conv = Conv2D(num_filters, kernel_size=(filter_size, embed_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    maxpool = MaxPool2D(pool_size=(seq_len - filter_size + 1, 1), strides=(1,1), padding='valid')(conv)
    maxpool_for_each_filter.append(maxpool)

# Concat all max pool layers
concatenate_maxpool = Concatenate(axis=1)(maxpool_for_each_filter)
flatten = Flatten()(concatenate_maxpool)
dropout = Dropout(drop_out)(flatten)
output = Dense(units=1, activation='sigmoid')(dropout)

model = Model(inputs=inputs, outputs=output) # Create model

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
# Save Vocabulary
vocab_file = checkpoint_dir + "/vocab.json"
save_vocab_json(vocab_file, vocabulary, params)

#checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=checkpoint_dir, histogram_freq=0)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

epochs = 20
batch_size = 32
verbose = 1
validation_split = 0.2
print("Traning Model...")
model.fit(x, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=[checkpoint, tensorboard_callback])

