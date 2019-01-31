#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:16:36 2018

@author: lovelace
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import keras
from keras.models import Sequential
from keras.optimizers import *
from keras.layers import *
import cv2

%matplotlib qt

# load dataset:
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test_copy = x_test.copy()
y_test_copy = y_test.copy()

# show example image:
num_images = 5
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.title(y_train[i])
    plt.imshow(x_train[i], cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
plt.show()

# why standardize?:
num_images = 5
stepsize = 1
for num_image in range(num_images):
    x, y, z = [], [], []
    for y_coord in range(0, 28, stepsize):
        for x_coord in range(0, 28, stepsize):
                x.append(x_coord)
                y.append(y_coord)
                z.append(x_train[num_image, x_coord, y_coord])
              
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap=plt.hot())
    plt.show()
    
# standardize data:
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0) + np.ones((28, 28)) * 1e-7

x_train = x_train - mean
x_test = x_test - mean
x_train = x_train / std
x_test = x_test / std

# why standardize? result:
stepsize = 1
for num_image in range(num_images):
    x, y, z = [], [], []
    for y_coord in range(0, 28, stepsize):
        for x_coord in range(0, 28, stepsize):
                x.append(x_coord)
                y.append(y_coord)
                z.append(x_train[num_image, x_coord, y_coord])
              
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap=plt.hot())
    plt.show()
    
# define ann:
# show link to cnn visualization
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile model:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# reshape because of technical reasons:
print('old x shape:', x_train.shape)
print('old y shape:', y_train.shape)
print('old y at index 0:', y_train[0])
print('old y at index 1:', y_train[1], '\n')
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
print('new x shape:', x_train.shape)
print('new y at index 0:', y_train[0])
print('new y at index 1:', y_train[1])
print('new y shape:', y_train.shape)

# for learning visualization:
from time import time
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# show tensorboard after training
model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[tensorboard])

# testing:
testing_scores = model.evaluate(x_test, y_test, batch_size=32)
print('testing accuracy:\t', round(testing_scores[1] * 100, 3), '%', 
      '\ntesting loss:\t\t', round(testing_scores[0], 3))
training_scores = model.evaluate(x_train, y_train, batch_size=32)
print('training accuracy:\t', round(training_scores[1] * 100, 3), '%', 
      '\ntraining loss:\t\t', round(training_scores[0], 3))

# view some predictions:
testing_indeces = np.random.randint(0, x_test_copy.shape[0], 9)
x_view = x_test[testing_indeces, :]
x_view_original = x_test_copy[testing_indeces, :]
y_view = y_test_copy[testing_indeces]
prediction = model.predict(x_view)
for i in range(3):
    for j in range(3):
        plt.subplot(33 * 10 + (i * 3) + j + 1)
        prediction_row = prediction[(i * 3) + j]
        prediction_row_max = np.max(prediction_row)
        predicted_label = prediction_row.argmax()
        title = str(predicted_label) + '  ' + str(round(prediction_row_max * 100, 2)) + '%'
        plt.title(title)
        plt.imshow(x_view_original[(i * 3) + j], cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])
plt.show()
