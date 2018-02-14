import theano
from theano import tensor as T
import numpy as np


###############################################
# EXAMPLE 1
###############################################

# declare two symbolic floating-point scalars
a = T.dscalar()
b = T.dscalar()
# create a simple expression
c = a + b
# convert the expression into a callable object that takes (a,b)
# values as input and computes a value for c
f = theano.function(inputs=[a,b], outputs=c)
# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
print f(1.5, 2.5)











###############################################
# EXAMPLE 2
###############################################

print('Building the model...')

x = T.dmatrix('x')  # Symbolic input matrix



# Initializing the weight matrix and bias vector
W = theano.shared(value=np.zeros((28*28, 10),dtype=theano.config.floatX), name='W')
b = theano.shared(value=np.zeros((10,),dtype=theano.config.floatX), name='b')


p_y_given_x = T.exp(T.dot(x, W) + b)
p_y_given_x = p_y_given_x / T.sum(p_y_given_x,axis=1)[:,None]



# Symbolic description of how to compute prediction as class whose
# probability is maximal
y_pred = T.argmax(p_y_given_x, axis=1)





print('Creating fake inputs...')
train_set_x = np.zeros((40, 28 * 28)).astype('float64')


# Compiling a Theano function `train_model` that returns the cost and the error of the training.
# Also, it updates the parameters of the model based on the rules defined in `updates`.
_model = theano.function(
        inputs=[x],
        outputs=y_pred)

tr = _model(train_set_x)
print tr, tr.shape


theano.printing.pydotprint(y_pred, outfile="logreg_pred_y.png", var_with_name_simple=True)