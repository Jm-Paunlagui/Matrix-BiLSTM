"""
@author: Jm-Paunlagui
@github: https://github.com/Jm-Paunlagui
@repository: https://github.com/Jm-Paunlagui/Matrix-BiLSTM
@description:
    This script is used to train a Bidirectional LSTM model for sentiment analysis.

    The model is trained on the CCC Evaluation comments of the students.

"""
import io

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pipeline import TextPreprocessing as MyTextPreprocessing
from pipeline import remove_stopwords

tf.keras.backend.clear_session()

# Read the imported data
df = pd.read_csv('dataset\\dataset-all-combined-final.csv')
print(df.head)

# Text Preprocessing and Cleaning
df.dropna(inplace=True,)
print(f"Number of rows: {df.shape[0]}")
df['sentence'] = df['sentence'].apply(lambda x: x.lower())
print(f"Lowercase: {df['sentence'].head(10)}")
df['sentence'] = df['sentence'].apply(lambda x: MyTextPreprocessing(x).clean_text())
print(f"Cleaned text: {df['sentence'].head(10)}")
# Removing stopwords
df['sentence'] = df['sentence'].apply(lambda x: remove_stopwords(x))
print(f"Removed stopwords: {df['sentence'].head(10)}")

# Shuffle the data and reset the index of the data frame to avoid any bias in the model training and testing process
df = df.sample(frac=1).reset_index(drop=True)
print(df.head(10))

# Parameters for the model
max_features = 6059
max_length = 300
epochs = 10
batch_size = 256
learning_rate = 3e-4

# Split the data into train and test
X = df['sentence']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(oov_token='<OOV>', lower=True, num_words=max_features)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Convert the text to sequences
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# Pad the sequences
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Create the model and compile it
model = Sequential([
    Embedding(max_features, 256, input_length=max_length, name="embedding"),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test))

# Evaluate the model
model.evaluate(X_test, y_test)

# Save the model
model.save('models\\model-backup.h5')

# Save the tokenizer
import pickle

with open('models/tokenizer-backup.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Metrics
predict_p = model.predict(X_test)
predict_p = predict_p.flatten()
pred = np.where(predict_p > 0.5, 1, 0)
classi_report = classification_report(y_test, pred)
confusionmatrix = confusion_matrix(y_test, pred)
accuracy = accuracy_score(y_test, pred)

print(f"Classification Report: \n {classi_report}")
print(f"Confusion Matrix: \n {confusionmatrix}")
print(f"Accuracy: {accuracy}")


# For visualization in Embedding Projector (https://projector.tensorflow.org/)
weights = model.get_layer('embedding').get_weights()[0]
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for word_num in range(1, max_features):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

