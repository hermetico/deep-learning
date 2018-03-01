
import os
# if you want to use Theano as backend
#os.environ['KERAS_BACKEND']='theano'
from keras.datasets import mnist
import random as rn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, MaxPool2D, Conv2D, concatenate, Add
from keras.models import Model
import tensorflow as tf
from keras.utils import np_utils, plot_model
from keras.callbacks import ModelCheckpoint




#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)
rn.seed(1234)


batch_size = 600
num_classes = 10
epochs = 3
filters = 10


# input image dimensions
img_rows, img_cols = 28, 28


# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('Printing the first 10 test examples')
for ii in range(10):
    plt.imshow(x_test[ii])
    plt.savefig('pic_{}.png'.format(ii))



# If using tensorflow as backend:
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train[:int(int(len(x_train)/batch_size)*batch_size)]
x_test = x_test[:int(int(len(x_test)/batch_size)*batch_size)]
y_train = y_train[:int(int(len(y_train)/batch_size)*batch_size)]
y_test = y_test[:int(int(len(y_test)/batch_size)*batch_size)]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # Makes it easier to process for the CNN
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)





def cnn(data_train = None,
        data_test = None ,
        batch_size = batch_size,
        epochs = epochs,
        train = True,
        predict = True,
        filters = filters):

    if data_train != None and data_test != None:
        print('Training and testing')
        x_train, y_train = data_train
        x_test, y_test = data_test # noramally, you don't need a y_test when testing

    elif data_test == None:
        print('Only training')
        x_train, y_train = data_train

    elif data_test == None:
        print('Only testing')
        x_test, y_test = data_test

    elif data_train == data_test == None:
        print('No data has been inserted')
        return


    ################################################
    #
    #   Define and connect the layer of the network
    #
    ################################################


    out = Dense(units=10, activation='softmax')(some_layer_you_have_defined_above)







    cnn = Model(inp, out)

    ################################################
    #
    #   Compile the model
    #
    ################################################

    cnn.summary()
    plot_model(cnn, to_file='I_<3_U_CNN.png')


    if train == True:

        ################################################
        #
        # Make something with the ModelCheckpoint function
        #
        ################################################

        cnn.fit(x_train, y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=#...#)



    if predict == True:
        print('Predicting...')
        print("Loading the model")
        cnn.load_weights(#....
        pred = cnn.predict(x_test, batch_size=batch_size)
        tmp = [np.argmax(i) for i in pred[:10]]
        print('The 10 first test list examples have been predicted as the following:')
        print(tmp)
        #score = cnn.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        #print(score)


if __name__ == "__main__":
    cnn((x_train, y_train), (x_test, y_test))
