#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:38:11 2023

@author: drewwest
"""

from tensorflow.keras.datasets import fashion_mnist

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#import matplotlib.pyplot as plt
#plot the first image in the dataset
#plt.imshow(X_train[0])

#reshape data to fit model
#image dimension 28x28x1
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

from tensorflow.keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
#create model
model = Sequential()
#add model layers
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
result = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

acc = result.history['accuracy'][-1]

print("Accuracy %.2f%%"%(acc*100))
#predict first 4 images in the test set
y_predict = model.predict(X_test[:4].astype(float))
print(y_predict)
#actual results for first 4 images in the test set
print(y_test)[:4]