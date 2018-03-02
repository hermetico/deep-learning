import os
# if you want to use Theano as backend
#os.environ['KERAS_BACKEND']='theano'

from keras.datasets import mnist
import random as rn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, Conv2D
from keras.models import Model
# The Keras functional API is the way to go for defining complex models,
# such as multi-output models, directed acyclic graphs, or models with shared layers.
import tensorflow as tf
from keras.utils import np_utils, plot_model
import keras

import sys
print("Python version: ", sys.version)
print("Keras version", keras.__version__)

# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# Making the results reproducible
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)
rn.seed(1234)


# Universal variables
batch_size = 600
num_classes = 10
epochs = 3
filters = 10# the depth of the convolutional volume


######################################################################


# input image dimensions
img_rows, img_cols = 28, 28

# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('Printing the first 10 test examples')
for ii in range(10):
    plt.imshow(x_test[ii])
    plt.savefig('pic_{}.png'.format(ii))


# reshaping the data so the keras network can understand it -> (rows, cols, channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Fitting the data to the batch size
# rounds the arrays to match the batch size
x_train = x_train[:int(int(len(x_train)/batch_size)*batch_size)]
x_test = x_test[:int(int(len(x_test)/batch_size)*batch_size)]
y_train = y_train[:int(int(len(y_train)/batch_size)*batch_size)]
y_test = y_test[:int(int(len(y_test)/batch_size)*batch_size)]

# Keras needs numpy arrays as types float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# values of pixels between 0 and 1
x_train /= 255 # Makes it easier to process for the CNN
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10) # Makes each true label into a onehot vector
y_test = np_utils.to_categorical(y_test, 10)



# How to shape the input:
# For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels)
# while "channels_first" assumes (channels, rows, cols).
# "channels_last" are default



######################################################################
# MAKING THE LAYERS OF THE NETWORK
####################################

# Connecting the layers of the network. Starting with an Input layer....
inp = Input(batch_shape=(batch_size,) + x_train.shape[1:])

# Making 2D convolutions with "filters" kernels with a shape of (2,2)
# Padding is valid, which means that no zero padding occurs.
con = Conv2D(filters, kernel_size=(2, 2),
             padding='valid', activation='relu', strides=1)(inp)


# You have to make sure that the shape is flattened before you can connect to an Dense layer.
flat = Flatten()(con)

# This is a softmax layer. Number of nodes has to be tha same as the number of classes
out = Dense(units=num_classes, activation='softmax')(flat)



######################################################################
# COMPILING THE NETWORK
############################

# Creating the Model - connecting the input layer with the output layer
cnn = Model(inp, out)
# Compiling the model - you define the learning algorithm and loss function.
# "metrics==['accuracy']" tells the model to print the accuracy whenever it updates.
cnn.compile(optimizer='sgd', loss='categorical_crossentropy',
            metrics=['accuracy'])
# Prints a summary of the model
cnn.summary()
# Plots the graph of teh model.
plot_model(cnn, to_file='I_<3_U_CNN.png')




######################################################################
# TRAINING AND PREDICTING
############################

# You can use validation_split to divide the training set into training and validation sets.
print("Training...")
cnn.fit(x_train, y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2) # part of the dataset for validating



print('Predicting...')
pred = cnn.predict(x_test, batch_size=batch_size)

tmp = [np.argmax(i) for i in pred[:10]]
print('The 10 first test list examples have been predicted as the following:')
for i, case in enumerate(y_test[:10]):
    print("prediction class %i --> %i prediction" % (np.argmax(case), tmp[i]))

print(tmp)









