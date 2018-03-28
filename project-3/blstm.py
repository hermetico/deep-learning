from __future__ import print_function
import os
#os.environ['KERAS_BACKEND']='theano'
#import theano.tensor as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Average, Dropout, Reshape, Flatten, Add, Maximum, Concatenate
from keras.layers import LSTM, Bidirectional
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import random as rn
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils




#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '7'
np.random.seed(1234)
tf.set_random_seed(1234)
rn.seed(1234)

#TODO doublecheck this
batch_size = 200
epochs = 50
N_LSTM = 100
VOCAB = ['a','c','g','t']
split_ratio = [0.7, 0.15, 0.15]


##########################################################
# DEFINING FUNCTION USED FOR PRE-PROCESSING OF INPUT DATA
##########################################################

def read_fasta_file_and_prepare_data(fasta_file):

    """
    This function extracts sequences from fasta files
    """
    seq_data = []
    ids = []
    with open(fasta_file,"r") as f:
        for line in f:
            line = line.strip().lower()
            if line[0] != ">":
                seq_data.append(line)
            else:
                tmp = line[::-1]
                tmp = tmp[:tmp.index('_')]
                tmp = tmp[::-1]
                ids.append(tmp)
    return seq_data, ids




def encode_seqs(input_a, input_b, vocab):
    """
    This function onehot embeds the sequences
    """

    inputs = input_a + input_b
    y_out = np.asarray([[1]] * len(input_a) + [[0]] * len(input_b))

    X_out = []
    for input in inputs:
        x = np.zeros((len(input),len(vocab)))

        for i in range(len(input)):
            x[i][vocab.index(input[i])] += 1
        X_out.append(x.flatten())

    X_out = np.asarray(X_out)
    X_out.reshape((len(inputs), 1, len(inputs[0])*len(vocab) )).astype('float32')
    return X_out, y_out




def split_data(seqs, ratios):
    """
    This function splits the input data into training, validation and test sets.
    """
    assert ratios[0] + ratios[1] < 1.0

    ntrain = int(round(ratios[0] * len(seqs)))
    nval = int(round(ratios[1] * len(seqs)))

    train = seqs[:ntrain]
    val = seqs[ntrain:ntrain+nval]
    test = seqs[ntrain+nval:]

    return train, val, test





def shuffle_seqs(seqs):
    """
    This function shuffles the inserted sequences and outputs a list
    with new shuffled sequences.
    """
    sf_sq = []
    for i in seqs:
        tmp = list(i)
        rn.shuffle(tmp)
        tmp = ''.join(tmp)
        sf_sq.append(tmp)

    return sf_sq


def output_to_line(layer_output, index, num_lstm):
    str_size = 100
    output = layer_output[index]
    return np.max(output, axis=1)



##########################################################
# IMPORTING DATA
##########################################################
print('Importing the FASTA file dna_TCCCACAAAC_2000_100.fa')
seqs, sids = read_fasta_file_and_prepare_data('dna_TCCCACAAAC_2000_100.fa')
bkg = shuffle_seqs(seqs)

print('Splitting the data into training, validation ang test set')
xtr, xva, xte = split_data(seqs, split_ratio)
bgtr, bgva, bgte = split_data(bkg, split_ratio)
trs, vas, tes = split_data(sids, split_ratio)

print('Encoding the data')
x_train, y_train = encode_seqs(xtr, bgtr, VOCAB)
x_val, y_val = encode_seqs(xva, bgva, VOCAB)
x_test, y_test = encode_seqs(xte, bgte, VOCAB)


print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_val shape', x_val.shape)
print('y_val shape', y_val.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)

##########################################################
# DEFINING THE NEURAL NETWORK
##########################################################

inp = Input(batch_shape=(batch_size,) + x_train.shape[1:])
inp_resh = Reshape((x_train.shape[1] / 4, 4))(inp)
inp_drop = Dropout(0.15)(inp_resh)
bi = Bidirectional(
    LSTM(N_LSTM,
        recurrent_dropout=0.25,
        return_sequences=True,
        kernel_regularizer='l2',
        recurrent_regularizer='l2'), name="BLSTM_layer")(inp_drop)

flat = Flatten()(bi)
out = Dense(1)(flat)



##########################################################
# Model
##########################################################


lstm_model = Model(inp, out)

##
##  SET OPTIMIZER AND LOSS FUNCTION AND COMPILE THE BAD BOY
##
adam = Adam(lr=0.009)
lstm_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
lstm_model.summary()



if __name__ == "__main__":

    ##########################################################
    # Train the model and save the base parameters
    ##########################################################

    model_name = "fancy_model.hdf5"
    mcp = ModelCheckpoint(model_name, save_best_only=True)

    hist = lstm_model.fit(x_train, y_train,
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data = [x_val, y_val],
                   callbacks=[mcp])


    # summarize history for accuracy
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.close('all')
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    plt.close('all')

    print('Predicting...')
    print("Loading the model")
    lstm_model.load_weights(model_name)
    pred = lstm_model.predict(x_test, batch_size=batch_size)
    score = lstm_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print(score)


    dna_profiles = lstm_model

    profiles = dna_profiles.predict(x_test, batch_size = batch_size)

    # extract information from the layer
    get_3rd_layer_output = K.function([dna_profiles.layers[0].input],
                                      [dna_profiles.get_layer("BLSTM_layer").output])
    blstm_layer = get_3rd_layer_output([x_test[:batch_size,:]])[0]

    for i in range(10):
        line = output_to_line(blstm_layer, i, N_LSTM)
        plt.plot(line)
        plt.xlabel('Base positions')
        plt.ylabel('Output value of LSTM layer')
        plt.axvline(int(tes[i]), color = 'black')
        plt.savefig("profiles_{}.png".format(i))
        plt.close('all')