
import os
# if you want to use Theano as backend
#os.environ['KERAS_BACKEND']='theano'
from keras.datasets import mnist
import random as rn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, MaxPool2D, Conv2D, Concatenate, Add
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





def assignment(data_train = None,
        data_test = None ,
        batch_size = batch_size,
        epochs = epochs,
        train = True,
        predict = True,
        filters = filters,
        model_name = "fancy_model.hdf5"):

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

    # Connecting the layers of the network. Starting with an Input layer....
    inp = Input(batch_shape=(batch_size,) + x_train.shape[1:])

    conv2D_1 = Conv2D(filters, kernel_size=(2, 2),
                 padding='same', activation='relu', strides=2)(inp)

    conv2D_2 = Conv2D(filters, kernel_size=(7, 7),
                 padding='valid', activation='relu')(inp)

    max_pool2D_1 = MaxPool2D(pool_size=(4, 4), padding='valid', strides=1)(conv2D_1)

    max_pool2D_2 = MaxPool2D(pool_size=(2, 2), padding='same')(conv2D_2)


    add_1 = Add()([max_pool2D_1, max_pool2D_2])

    conv2D_3 = Conv2D(filters, kernel_size=(2, 2),
                      padding='valid', activation='relu', strides=1)(add_1)

    flat = Flatten()(conv2D_3)

    dense_1 = Dense(units=1024, activation="elu")(flat)
    dense_2 = Dense(units=512, activation="selu")(dense_1)

    con_1 = Concatenate()([dense_2, flat])

    out = Dense(units=10, activation='softmax')(con_1)



    cnn = Model(inp, out)

    ################################################
    #
    #   Compile the model
    #
    ################################################
    cnn.compile(optimizer='sgd', loss='categorical_crossentropy',
                metrics=['accuracy'])

    cnn.summary()
    plot_model(cnn, to_file='I_<3_U_CNN.png')


    if train:

        ################################################
        #
        # Make something with the ModelCheckpoint function
        #
        ################################################
        # TODO improve the checkpoint
        checkpoint = ModelCheckpoint(model_name, save_best_only=True)

        cnn.fit(x_train, y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[checkpoint])



    if predict:
        print('Predicting...')
        print("Loading the model")
        cnn.load_weights(model_name)
        pred = cnn.predict(x_test, batch_size=batch_size)
        tmp = [np.argmax(i) for i in pred[:10]]
        print('The 10 first test list examples have been predicted as the following:')
        for i, case in enumerate(y_test[:10]):
            print("prediction class %i --> %i prediction" % (np.argmax(case), tmp[i]))
        print(tmp)
        score = cnn.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        print(score)


if __name__ == "__main__":
    assignment((x_train, y_train), (x_test, y_test))
