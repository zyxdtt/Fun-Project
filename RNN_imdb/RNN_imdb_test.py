from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
model=load_model('RNN_MODEL.h5')
word_to_index=imdb.get_word_index()
text=input('Enter a regard no more than 200 words:')
text=text.lower()
word_list=text.split()
word_per_review=200
vec=[1,]
for word in word_list:
    if word in word_to_index:
        index=word_to_index[word]
        if index<=10000:
            vec+=[index+3]
        else:
            vec+=[2]
    else:
        vec+=[2]
vec=pad_sequences([vec],maxlen=word_per_review)
result=model.predict(vec)
print(result)