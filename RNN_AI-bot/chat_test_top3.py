from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np

print('Chat-bot preparing...')
model=load_model('best_model.h5')
word_list=dict()
max_len=50
max_knowledge=10000
EOU_ID=3

with open('word_list.txt',mode='r',encoding='utf-8') as word_dict:
    turns=0
    for record in word_dict:
        word,idx=record.split()
        word_list[word]=int(idx)
        turns+=1
        if turns>max_knowledge:
            break

#Squint 1
id_to_word={num+1:word for word,num in word_list.items()}

print()
initial='Oh !'#Sure to have Oh and !
print(initial)
history=initial.split()
history=[word_list[word]+1 for word in history]#Squint 1
history+=[EOU_ID]

def generate_machine(seq,model,max_len):
    padded=pad_sequences([seq],maxlen=max_len,padding='post')
    pred=model.predict(padded,verbose=0)
    top3_idx=np.argpartition(pred[0],-3)[-3:]#Change from greedy to top3
    random_idx=np.random.choice(top3_idx)
    return random_idx

while True:
    response=input()
    words=re.findall(r'\w+|[^\w\s]',response)
    words=[word_list[word]+1 if word in word_list else 1 for word in words]#Squint 1
    history+=words
    history+=[EOU_ID]
    history=history[-50:]
    next_word=0
    while next_word!=EOU_ID:
        next_word=generate_machine(history,model,max_len)
        history+=[next_word]
        if next_word==1:
            print('?',end=' ')
        elif next_word==EOU_ID:
            print()
        else:
            print(id_to_word[next_word],end=' ')
        history=history[-50:]
