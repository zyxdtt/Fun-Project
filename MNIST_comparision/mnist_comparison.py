from lzma import PRESET_DEFAULT
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import netron

input('Loading data...')
(X_train,y_train),(X_test,y_test)=mnist.load_data()
# Keep raw label copies for classical ML (we will convert to categorical later for Keras)
y_train_raw = y_train.copy()
y_test_raw = y_test.copy()

# Create flattened versions of the image data for scikit-learn models and TSNE
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

input('Let\'s see the shape of our data')
input(f'X_train shape: {X_train.shape}\n\
y_train shape: {y_train.shape}\n\
X_test shape: {X_test.shape}\n\
y_test shape: {y_test.shape}')

input("Let's see some picture of our data")
input('There are 24 photos: ')
sns.set(font_scale=2)
choice=np.random.choice(np.arange(len(X_train)),24,replace=False)
figure,axes=plt.subplots(nrows=4,ncols=6,figsize=(16,9))
for item in zip(axes.ravel(),X_train[choice],y_train[choice]):
    axes,image,target=item
    axes.imshow(image,cmap=plt.cm.gray_r)#We want gray image
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()
plt.show()

input("Before our formal ML")
input("Let's see weather 2D visualize can finish the task")
tsne=TSNE(n_components=2,random_state=11)
X_small=X_train_flat[:2000]
X_small=X_small.reshape(2000,-1)    
y_small=y_train_raw[:2000]
reduced_data=tsne.fit_transform(X_small)

input(f'Now reduced_data is a 2D Tensor,shape: {reduced_data.shape}')
input("We can show it on figure directly")
dots=plt.scatter(reduced_data[:,0],reduced_data[:,1],c=y_small,
                 cmap=plt.cm.get_cmap('nipy_spectral_r',10))
colorbar=plt.colorbar(dots)
plt.show()

input("Let's begin our machine learning program")
input("First we use KNeighborClassifier to execute our task.")
knn=KNeighborsClassifier()

input("Training the model...")
knn.fit(X=X_train_flat,y=y_train_raw)

input('Predict the test dataset...')
predicted=knn.predict(X=X_test_flat)
expected=y_test_raw
wrong=[(p,e) for (p,e) in zip(predicted,expected) if p!=e]
print(f'We can see that percentage of correct predictions: '
      f'{(len(expected)-len(wrong))/len(expected):.2%}')

input("Let's see why our model output bad answers")
input("We wiil give 5 examples")
print(wrong[:5])

input("We can also use a confusion matrix to see our predictions")
confusion=confusion_matrix(y_true=expected,y_pred=predicted)
print(confusion)

input("Generate a classification report")
names=[str(i) for i in range(10)]
print(classification_report(expected,predicted, target_names=names))

input("Heat map")
df=pd.DataFrame(confusion,index=range(10),columns=range(10))
ax = sns.heatmap(df, annot=True, cmap='nipy_spectral_r')
plt.tight_layout()
plt.show()

input("Run multiple models to decide best")
input("We add SVC and Bayes classifier")
estimators={"KNeighborsClassifier":knn,
            'SVC':SVC(),
            'GaussianNB':GaussianNB()}
for name,obj in estimators.items():
    # Use 1800 samples from training set for k-fold cross-validation to speed up
    X_subset = X_train_flat[:1800]
    y_subset = y_train_raw[:1800]
    kfold = KFold(n_splits=5, random_state=11, shuffle=True)
    scores = cross_val_score(estimator=obj, X=X_subset, y=y_subset, cv=kfold, scoring='accuracy')
    print(f'{name:>20}: mean accuracy={scores.mean():.2%}; standard deviation={scores.std():.2%}')

input("Keras want 3D Tensor to process a picture")
input("So we must add another dimension to our data")
X_train=X_train.reshape((60000,28,28,1))
X_test=X_test.reshape((10000,28,28,1))
input(f'After adding dimension:\n\
        X_train shape: {X_train.shape}\n\
        X_test shape: {X_test.shape}')

#We also have to normalization our data
X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

#Then,one-hot encoding
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

input("Let's begin to configure a CNN")
cnn=Sequential()
input('First layer:64 filters,3*3 kernel size,relu function')
cnn.add(Conv2D(filters=64,kernel_size=(3,3),
               activation='relu',input_shape=(28,28,1)))
input("Now we have 26*26*64 features")
input("So we should pooling to decrease")
input("We use MaxPooling,you can also use dropout")
input('MaxPooling size:2*2')
cnn.add(MaxPooling2D(pool_size=(2,2)))
input("13*13*64 features")
input('Second layer:128 filters,3*3 kernel size,relu function')
cnn.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
input("11*11*128 features")
cnn.add(MaxPooling2D(pool_size=(2,2)))
input("5*5*128 features")
cnn.add(Flatten())
input("1*3200 features")
input("Then add a Dense layer to decrease feature-number")
cnn.add(Dense(units=128,activation='relu'))
input("Last,transfer to 10 units to output the possibility")
cnn.add(Dense(units=10,activation='softmax'))

input("Let's see our CNN")
input(f'Our model:\n{cnn.summary()}')

input('CNN has been successfully created!\nSaving model...')
input("Let's use a beautiful interactive tools to show our model")
cnn.save('mnist_cnn_model_before_fitting.h5')
netron.start('mnist_cnn_model_before_fitting.h5')

cnn.compile(optimizer='adam',loss='categorical_crossentropy',
            metrics=['accuracy'])

input('Training the model...')
cnn.fit(X_train,y_train,epochs=5,batch_size=64,
        validation_split=0.1)

input('Evaluating model...')
loss,accuracy=cnn.evaluate(X_test,y_test)

input('See some bad predictions: ')
predictions=cnn.predict(X_test)
images=X_test.reshape((10000,28,28))
incorrect_predictions=[]
for i,(p,e) in enumerate(zip(predictions,y_test)):
    predicted,expected=np.argmax(p),np.argmax(e)
    if predicted!=expected:
        incorrect_predictions.append((i,images[i],predicted,expected))
input(f'The number of wrong predictions in 10,000 tests: {len(incorrect_predictions)}')
input('See some of 24 incorrect pictures: ')
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 12))
axes = axes.ravel()

num_to_show = min(24, len(incorrect_predictions))
for i in range(num_to_show):
    idx, image, predicted, expected = incorrect_predictions[i]
    axes[i].imshow(image, cmap=plt.cm.gray_r)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(f'predict: {predicted}\nexpected: {expected}')

for i in range(num_to_show, 24):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

input('See why our model gave wrong predictions')
input('We will give two examples: ')
cnt=0
incorrect_probabilities=[]
for i, (p, e) in enumerate(zip(predictions, y_test)):
    predicted,expected=np.argmax(p),np.argmax(e)
    if predicted!=expected:
        incorrect_probabilities.append((p,expected))
        cnt+=1
    if cnt==2:
        break
for (p,expected) in incorrect_probabilities:
    print(f'expected: {expected}')
    for index,probabilities in enumerate(p):
        print(f'{index}: {probabilities:.4f}')
    print()

input('Task done! Save the final model')
cnn.save('mnist_cnn.h5')
