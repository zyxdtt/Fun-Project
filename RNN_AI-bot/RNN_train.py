from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from random import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

multi_text=[]
number_of_words=10000

with open('text_index.txt',mode='r') as text:
    all_nums=[int(x) for x in text.read().split()]
    tex=[]
    for num in all_nums:
        if num==0:
            multi_text.append(tex)
            tex=[]
        else:
            tex.append(num)
    if tex:
        multi_text.append(tex)

def process_word(num):
    if num>number_of_words:
        return 1
    else:
        return num+1

multi_text=[[process_word(num) for num in seq] for seq in multi_text]
shuffle(multi_text)
split_idx=int(len(multi_text)*0.9)
train=multi_text[:split_idx]
val=multi_text[split_idx:]

def create_generator(sequences):
    for seq in sequences:
        seq=seq[:51]
        for i in range(1,len(seq)):
            X=seq[:i]
            y=seq[i]
            yield X,y

def create_dataset(sequences,max_len,batch_size):
    dataset=tf.data.Dataset.from_generator(
        lambda:create_generator(sequences),
        output_types=(tf.int32,tf.int32),
        output_shapes=([None],[])
        )
    dataset=dataset.padded_batch(
        batch_size,padded_shapes=([max_len],[]),
        padding_values=(0,0)
        )
    return dataset.repeat()

batch_size=128#Change from 64 to 128
word_per_par=50#change from 200 to 50

train_dataset=create_dataset(train,word_per_par,batch_size)
val_dataset=create_dataset(val,word_per_par,batch_size)

total_train_samples=sum(len(seq)-1 for seq in train)
total_val_samples=sum(len(seq)-1 for seq in val)
#Here I use floor,which will discard some data
#If you do not like,you can use ceil
steps_per_epoch=total_train_samples//batch_size
validation_steps=total_val_samples//batch_size

rnn=Sequential()
rnn.add(Embedding(input_dim=number_of_words+2,
                  output_dim=128,input_length=word_per_par))
rnn.add(LSTM(units=256,dropout=0.2,recurrent_dropout=0.2))#Units from 512 to 256
rnn.add(Dense(units=number_of_words+2,activation='softmax'))
rnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',metrics=['accuracy'])

callbacks=[
    ModelCheckpoint(
        'best_model.h5',monitor='val_loss',
        save_best_only=True,verbose=1),
    EarlyStopping(monitor='val_loss',patience=3,
                  restore_best_weights=True,verbose=1)
    ]

history=rnn.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=callbacks,
    verbose=1)
