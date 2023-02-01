"""
@author: Jm-Paunlagui
@github: https://github.com/Jm-Paunlagui
@repository:
@description:
    This script is used to train a Bidirectional LSTM model for sentiment analysis.

    The model is trained on the CCC Evaluation comments of the students.

"""

import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.initializers.initializers_v2 import RandomUniform, Orthogonal

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout, GlobalMaxPooling1D, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

# Read the preprocessed data
df = pd.read_csv('dataset\\datasets-labeled-cleaned.csv')

# Parameters for the model
max_features = 800
max_length = 300
epochs = 10
batch_size = 128
learning_rate = 3e-4

# Split the data into train and test
X = df['sentence']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

# Tokenize the text
tokenizer = Tokenizer(oov_token='<OOV>', lower=True, num_words=max_features)
tokenizer.fit_on_texts(X_train)

# Convert the text to sequences
X_train = tokenizer.texts_to_sequences(X_train)

# Pad the sequences
X_train = pad_sequences(X_train, maxlen=max_length)


def bidirectional_model(input_shape):
    """
    :param input_shape:
    :return: deep learning model for sentiment analysis

    Model Architecture:
    Embedding -> Bidirectional LSTM -> Global Max Pooling -> Dense -> Dropout -> Dense

    Embedding:
    - input_dim: size of the vocabulary
    - output_dim: size of the dense vector to represent each token
    - input_length: length of the input sequence

    Bidirectional LSTM:
    - units: number of units in the LSTM
    - return_sequences: whether to return the last output in the output sequence, or the full sequence
    - activation: activation function to use
    - recurrent_activation: activation function to use for the recurrent step
    - kernel_initializer: initializer for the kernel weights matrix
    - recurrent_initializer: initializer for the recurrent kernel weights matrix
    - bias_initializer: initializer for the bias vector
    - unit_forget_bias: if True, add 1 to the bias of the forget gate at initialization

    Global Max Pooling:
    - pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal).
    - strides: Integer, tuple of 2 integers, or None. Strides values.
    - padding: One of "valid" or "same" (case-insensitive).

    Dense:
    - units: Positive integer, dimensionality of the output space.
    - activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - use_bias: Boolean, whether the layer uses a bias vector.
    - kernel_initializer: Initializer for the kernel weights matrix.
    - bias_initializer: Initializer for the bias vector.

    Dropout:
    - rate: Float between 0 and 1. Fraction of the input units to drop.
    - noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
    - seed: A Python integer to use as random seed.
    """
    inputs = Input(shape=input_shape, dtype='int32')

    embedding = Embedding(
        input_dim=800, output_dim=256, embeddings_initializer=RandomUniform(
            minval=-0.05, maxval=0.05
        ), input_length=None, name='embedding_1'
    )(inputs)

    bidirectional = Bidirectional(
        LSTM(
            units=64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid',
            kernel_initializer=glorot_uniform(),
            recurrent_initializer=Orthogonal(gain=1.0), bias_initializer='zeros',
            unit_forget_bias=True), merge_mode='concat', name='bidirectional'
    )(embedding)

    global_max_pooling1d = GlobalMaxPooling1D(
        name='global_max_pooling1d'
    )(bidirectional)

    dense = Dense(
        units=64, activation='relu', kernel_initializer=glorot_uniform(),
        bias_initializer='zeros', name='dense'
    )(global_max_pooling1d)

    dropout = Dropout(
        0.5, name='dropout'
    )(dense)

    outputs = Dense(
        1, activation='sigmoid', kernel_initializer=glorot_uniform(), bias_initializer='zeros',
        name='output_1'
    )(dropout)

    return Model(inputs=inputs, outputs=outputs, name='model')


model = bidirectional_model((max_length,))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length)
model.evaluate(X_test, y_test)

# Save the model
model.save('models\\bilstm.h5')

# Save the tokenizer
import pickle

with open('models\\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)








