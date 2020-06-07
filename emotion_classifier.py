# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:08:06 2020

@author: aditya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_text import CleanText
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model, Sequential
from keras.layers import LSTM
from keras.layers import Flatten, Dense, Dropout, Activation, Input ,BatchNormalization
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


tweets = pd.read_csv('Tweets.csv')
tweets = tweets[['text','airline_sentiment']]

clean = CleanText()

tweets['text'] = tweets['text'].apply(lambda x: clean.clean(x))

docs = tweets['text']
labels = tweets['airline_sentiment']
le = LabelEncoder()
labels_en = le.fit_transform(labels) #Neutral: 1, Positive: 2, Negative: 0
labels_en = keras.utils.to_categorical(np.asarray(labels_en))

#tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
#encode the documents
encoded_docs = t.texts_to_sequences(docs)

# Function to find length of the Longest Sentence
def maxLength(sentence):
    max_length = 0
    for i in sentence:
        length = len(i)
        if length > max_length:
            max_length = length
    return max_length

max_length = 72

#pad docs to max length
padded_docs = pad_sequences(encoded_docs, maxlen = 72, padding = 'post')

# Train/Dev/Test set: 80/10/10 split

xtrain, xdev, ytrain, ydev = train_test_split(padded_docs, labels_en, test_size = 0.2, random_state = 8)

# xtest and ytest is our Unseen Data which will be used to get an Unbiased Evaluation of the Model
xdev, xtest, ydev, ytest = train_test_split(xdev, ydev, test_size = 0.5, random_state = 8)


#load embedding into memory
embeddings_index = dict()
f = open('Glove/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()    

# Weight matrix for words in training
embedding_matrix = np.zeros((vocab_size , 100))    
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Model
embedding_layer = Embedding(vocab_size,
                            100,
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable = False)

sequence_input = Input(shape = (max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(60, return_sequences = True, recurrent_dropout = 0.5)(embedded_sequences)
x = Dropout(0.4)(x)
x = LSTM(60, return_sequences = True, recurrent_dropout = 0.5)(x)
x = Dropout(0.4)(x)
x = LSTM(60, return_sequences = True, recurrent_dropout = 0.5)(x)
x = Dropout(0.4)(x)
x = LSTM(60, return_sequences = True, recurrent_dropout = 0.5)(x)
x = Dropout(0.4)(x)
x = LSTM(60, return_sequences = True, recurrent_dropout = 0.5)(x)
x = Dropout(0.4)(x)
x = LSTM(60, recurrent_dropout = 0.5)(x)
x = Dropout(0.4)(x)
x = Dense(60, activation = 'relu')(x)
x = Dropout(0.4)(x)
preds = Dense(3 , activation = 'softmax')(x)

model = Model(sequence_input, preds)
opt = Adam(learning_rate = 0.01, beta_1 = 0.9 , beta_2 = 0.999, decay = 0.005)
model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

history = model.fit(xtrain, ytrain,
                    batch_size = 64,
                    epochs = 20,
                    shuffle = True,
                    workers = 6,
                    validation_data = (xdev, ydev))


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#model.save('sentiment_classifier4.h5')

# Test your final model on the test set to get an unbiased estimate
test = keras.models.load_model('sentiment_classifier4.h5')
loss, acc = test.evaluate(xtest, ytest)
print(acc)
preds = test.predict(xtest)

# Final Evaluation: Train set: 86%, Dev Set: 81%, Test Set: 80%





















