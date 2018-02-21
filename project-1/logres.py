from __future__ import print_function
import cPickle
import gzip
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt



BATCH_SIZE=60
DATASET='mnist.pkl.gz'
ETA=0.15
N_EPOCHS=20
EARLY_STOPPING = 10



def load_data(dataset):

    # LOAD DATA
    # Can download MNIST data from: http://deeplearning.net/tutorial/gettingstarted.html#gettingstarted

    print('Loading the MNIST data...')
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    print("length of training set: ", len(train_set[0]))
    print("length of validation set: ", len(valid_set[0]))
    print("length of test set: ", len(test_set[0]))

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval





def logreg(n_in=28 * 28, n_out=10, batch_size=60, dataset='mnist.pkl.gz', learning_rate=0.13, n_epochs=10, early_stopping = 5):


    # Loading the data
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size




    # Building the model with symbolic matrices and vectors
    print('Building the model...')
    x = T.matrix('x')  # Symbolic input matrix
    y = T.ivector('y')  # Symbolic vector for the labels


    # Initializing the weight matrix and bias vector
    W = theano.shared(value=np.zeros((n_in, n_out),dtype=theano.config.floatX), name='W', borrow=True)
    b = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX), name='b',borrow=True)


    # Symbolic expression for computing the matrix of class-membership probabilities
    # Where:
    # W is a matrix where column-k represent the separation hyperplane for class-k
    # x is a matrix where row-j  represents input training sample-j
    # b is a vector where element-k represent the free parameter of hyperplane-k
    p_y_given_x = T.exp(T.dot(x, W) + b)
    p_y_given_x = p_y_given_x / T.sum(p_y_given_x,axis=1)[:,None]

    ## using softmax implementation
    #p_y_given_x = T.nnet.softmax(T.dot(x, W) + b) # <-- this is another way to calculate predictions


    # Symbolic description of how to compute prediction as class whose
    # probability is maximal
    y_pred = T.argmax(p_y_given_x, axis=1)


    # Allocate symbolic variables for the data.
    # This is used for iterating through the mini-batches
    index = T.lscalar()


    # Calculating the errors, is y_pred == y ?
    # the T.neq operator returns a vector of 0s and 1s, where 1
    # represents a mistake in prediction
    errors = T.mean(T.neq(y_pred, y))


    # y.shape[0] is (symbolically) the number of rows in y, i.e.,
    # number of examples (call it n) in the minibatch
    # T.arange(y.shape[0]) is a symbolic vector which will contain
    # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
    # Log-Probabilities (call it LP) with one row per example and
    # one column per class LP[T.arange(y.shape[0]),y] is a vector
    # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
    # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
    # the mean (across minibatch examples) of the elements in v,
    # i.e., the mean log-likelihood across the minibatch.
    cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])


    # compute the gradient of cost with respect to W and b
    # --> in which direction should the parameters go
    # in order to improve the predictions of the training data
    g_W = T.grad(cost=cost, wrt=W)
    g_b = T.grad(cost=cost, wrt=b)


    # Learning_rate dictes how much the parameters should be changed
    # after every mini-batch
    updates = [(W, W - learning_rate * g_W),
               (b, b - learning_rate * g_b)]






    # Compiling a Theano function `train_model` that returns the cost and the error of the training.
    # Also, it updates the parameters of the model based on the rules defined in `updates`.
    train_model = theano.function(
        inputs=[index],
        outputs=[cost, errors],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )



    ########################################
    #
    # Compile the validation function
    # Write a validation function that outputs the errors, using the validation data
    # See the theano.function called "train_model"
    #
    ########################################
    validate_model = theano.function(
        inputs=[index],
        outputs=[errors],
        updates=updates,
        givens={
            x: valid_set_x[index * batch_size : (index + 1) * batch_size],
            y: valid_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )





    ########################################
    #
    # Compile the test function
    # Write a test function that outputs the errors, using the test data
    # See the theano.function called "train_model"
    #
    ########################################
    test_model = theano.function(
        inputs=[index],
        outputs=[errors],
        updates=updates,
        givens={
            x: test_set_x[index * batch_size : (index + 1) * batch_size],
            y: test_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )





    ########################################
    # Training the model

    print('Training the model...')

    best_validation_loss = 1
    start_time = timeit.default_timer()

    keep_looping = True
    stop = 0
    epoch = 0
    while epoch < n_epochs and keep_looping:

    #for epoch in range(n_epochs):

        # Running mini batches through the training function
        mean_tr_error = []
        # training the model
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost, train_error = train_model(minibatch_index)
            # iteration number
            #iter = (epoch - 1) * n_train_batches + minibatch_index
            mean_tr_error.append(train_error)
        
        current_train_error = np.mean(mean_tr_error)
        print('\nepoch %i, minibatch %i/%i, training error %f %%' %
                (epoch +1,
                minibatch_index + 1,
                n_train_batches,
                current_train_error * 100.))


        ########################################################
        #
        # RUN THE MINI-BATCHES THROUGH THE VALIDATION FUNCTION
        #
        # USE A VALIDATION FUNCTION TO VALIDATE THE "n_valid_batches"
        # PRINT THE VALIDATION ERROR FOR EVERY EPOCH
        #
        ########################################################
        mean_val_error = []
        for minibatch_index in range(n_valid_batches):
            mean_val_error.append(validate_model(minibatch_index))

        current_val_error = np.mean(mean_val_error)
        print('\n\tepoch %i, minibatch %i/%i, validation error %f %%' %
              (epoch + 1,
               minibatch_index + 1,
               n_valid_batches,
               current_val_error * 100.))

        # if we got the best validation score until now
        if current_val_error < best_validation_loss:
            best_validation_loss = current_val_error
            # test it on the test set


            ########################################################
            #
            # RUN THE MINI-BATCHES THROUGH THE TEST FUNCTION
            #
            # USE A TEST FUNCTION TO VALIDATE THE "n_test_batches"
            # PRINT THE TEST ERROR FOR EVERY EPOCH
            #
            ########################################################
            mean_test_error = []
            for minibatch_index in range(n_test_batches):
                mean_test_error.append(test_model(minibatch_index))

            current_test_error = np.mean(mean_val_error)
            print('\nTESTING: epoch %i, minibatch %i/%i, test error %f %%' %
                  (epoch + 1,
                   minibatch_index + 1,
                   n_valid_batches,
                   current_test_error * 100.))

            # retreive the best parameters
            params = [W, b]
            saved_params = {'W': W, 'b': b}

            ########################################################
            #
            # SAVE THE BEST PARAMETERS
            #
            ########################################################
            save_file = open('best_model.pkl', 'wb')  # this will overwrite current contents
            cPickle.dump(saved_params, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL

            save_file.close()
            stop = 0

        else:
            stop += 1
            ########################################################
            #
            # IMPLEMENT EARLY STOPPING THAT STOPS THE LOOP/TRAINING
            # IF THE MODEL IS NO LONGER PROGRESSING
            #
            ########################################################
            if stop <= early_stopping:
                keep_looping = False
                print("Early stoping")

        # next expoch
        epoch += 1

    end_time = timeit.default_timer()
    print(
        (
            '\nOptimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., current_test_error * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)



def predict():

    # load the saved model
    params = cPickle.load(open('best_model.pkl'))

    print('\nBuilding the predicter...')


    ########################################################
    #
    # REBUILD THE LOGISTIC REGRESSION NETWORK USING SYMBOLIC VARIABLES
    # USE THE LOADED PARAMETERS TO INITIALIZE THE WEIGHT MATRIX AND THE BIAS VECTOR
    #
    ########################################################
    x = T.matrix('x')  # Symbolic input matrix

    p_y_given_x = T.nnet.softmax(T.dot(x, params['W']) + params['b']) # <-- this is another way to calculate predictions
    y_pred = T.argmax(p_y_given_x, axis=1)

    # Symbolic description of how to compute prediction as class whose
    # probability is maximal
    y_pred = T.argmax(p_y_given_x, axis=1)

    # Compile a predictor function
    predict_model = theano.function(
        inputs=[x],
        outputs=y_pred)


    # Printing a graph of the predicting network
    theano.printing.pydotprint(y_pred, outfile="logreg_predicter.png", var_with_name_simple=True)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("\n\nPredicted values for the first 10 examples in test set:")
    print(predicted_values)

    for i,j in enumerate(test_set_x[:10]):
        plt.imshow(j.reshape((28,28)))
        plt.savefig("fig_{}.png".format(i))




if __name__ == '__main__':
    logreg(batch_size=BATCH_SIZE, dataset=DATASET, learning_rate=ETA, n_epochs=N_EPOCHS, early_stopping=EARLY_STOPPING)
    predict()

