from tabnanny import verbose
from sklearn import metrics
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
import numpy as np

number_of_words = 10000
words_per_review = 200
epochs = 10
batch_size = 32

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)
X_add=X_test[:5000]# cut 5000 sets from test to train
y_add=y_test[:5000]
X_train=np.concatenate([X_train,X_add])
y_train=np.concatenate([y_train,y_add])
X_test=X_test[5000:]
y_test=y_test[5000:]

X_train = pad_sequences(X_train, maxlen=words_per_review)
X_test = pad_sequences(X_test, maxlen=words_per_review)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=11, test_size=0.10)

# set model
rnn = Sequential()
rnn.add(Embedding(input_dim=number_of_words, output_dim=128, input_length=words_per_review))
rnn.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
rnn.add(Dense(units=1, activation='sigmoid'))

rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(rnn.summary())

# train model
history = rnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2)

results = rnn.evaluate(X_test, y_test)
print(f"test loss: {results[0]}, test accuracy: {results[1]}")

rnn.save('RNN_MODEL.h5')