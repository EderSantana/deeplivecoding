
# coding: utf-8

## Theano Basics

# In[1]:

from __future__ import print_function

import theano
import numpy as np

from theano import tensor as T
floatX = theano.config.floatX


# In[2]:

# Convention:
#  uppercase: symbolic theano element or function
#  lowercase: numpy array
W = T.vector('w')
X = T.matrix('X')
Y = X.dot(W)
F = theano.function([W,X], Y)

w = np.ones(4)
x = np.ones((10,4))
y = F(w,x)
print(y)


# In[3]:

# The most underused tool in machine learning
# AUTODIFF
grad_w = T.grad(Y.sum(), W)
F_grad = theano.function([W,X], grad_w)
g = F_grad(w,x)
# this should be equal to the sum of the columns of X (do you know how to matrix calculus?)
print(g)


# In[4]:

# An easier example
B = T.scalar('E')
R = T.sqr(B)
A = T.grad(R, B)
Z = theano.function([B], A)
i = 2
l = Z(i)
print(l)


# In[5]:

# If that didn't blow your mind, well, it should have.
def sharedX(X):
    return theano.shared(X.astype(floatX))

B = sharedX(np.ones(2))
R = T.sqr(B).sum()
A = T.grad(R, B)
Z = theano.function([], R, updates={B: B - .1*A})
for i in range(10):
    print('cost function = {}'.format(Z()))
    print('parameters    = {}'.format(B.get_value()))
# Try to change range to 100 to see what happens


## Neural Nets

# In[6]:

""" Now that we now how to sum, we have enough to Deep Learn
 ... I should say something in the board about the Model-View-Controller way we usually
     deep learn with Theano.
     Model      : Neural net parameters and dataset generator
     View       : Logging, graph updates, saving cross-validated best parameters
     Controller : Update algorithm that follows gradient directions to optimize paramters

 Download this dataset : http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
 
"""
get_ipython().magic(u'matplotlib inline')
import cPickle
from pylab import imshow

train_set, valid_set, test_set = cPickle.load(file('mnist.pkl', 'r'))
print(len(train_set))
train_x, train_y = train_set
test_x , test_y  = test_set
print(train_x.shape)
print(train_y.shape)
_ = imshow(train_x[0].reshape((28,28)), cmap='gray')


# In[7]:

def batch_iterator(x, y, batch_size):
    num_batches = x.shape[0] // batch_size
    for i in xrange(0,num_batches):
        # TODO: use random integers instead of consecutive
        #   values to avoid biased gradients
        first = i * batch_size
        last  = (i+1) * batch_size
        x_batch = x[first:last].astype(floatX)
        y_pre   = y[first:last]
        y_batch = np.zeros((batch_size, 10))
        for row, col in enumerate(y_pre):
            y_batch[row, col] = 1
        yield (x_batch, y_batch.astype(floatX))

for x,y in batch_iterator(train_x, train_y, 10000):
    print('{}, {}'.format(x.shape, y.shape))
print(y[0])
_ = imshow(x[0].reshape((28,28)), cmap='gray')


# In[13]:

# Define layers
def rectifier(input_dim, output_dim, X):
    W = sharedX(np.random.normal(0, .001, size=(input_dim, output_dim)))
    b = sharedX(np.zeros((output_dim,)))
    Z = T.dot(X,W) + b.dimshuffle('x',0)
    O = T.switch(Z>0, Z, 0)
    return W,b,O

def softmax(input_dim, output_dim, X, Y):
    W = sharedX(np.random.normal(0, .001, size=(input_dim, output_dim)))
    b = sharedX(np.zeros((output_dim,)))
    Z = T.dot(X,W) + b.dimshuffle('x',0)
    O = T.nnet.softmax(Z)
    cost = T.nnet.binary_crossentropy(O, Y).sum(axis=-1).mean()
    return W,b,O,cost

X = T.matrix('X')
Y = T.matrix('Y')
W0, b0, O0 = rectifier(784, 100, X)
W1, b1, O1 = rectifier(100, 100, O0)
W2, b2, O2, cost = softmax(100, 10,  O1, Y)

# Always write tests
F = theano.function([X,Y], [cost, O2])
x = np.zeros((100,784)).astype(floatX)
y = np.ones((100,10)).astype(floatX)
c, z = F(x,y)
assert c>0
assert z.shape == (100,10)
print(z[0])


# In[14]:

from collections import OrderedDict
params = [W0, b0, W1, b1, W2, b2]
updates = dict()
for p in params:
    updates[p] = p - .01 * T.grad(cost, p)
updates = OrderedDict(updates)
trainer = theano.function([X,Y], cost, updates=updates)


# In[15]:

num_epochs = 100
for i in range(num_epochs):
    print('-'*10)
    print('Epoch: {}'.format(i))
    for iter,b in enumerate(batch_iterator(train_x, train_y, 128)):
        x = b[0]
        y = b[1]
        last_cost = trainer(x,y)
    print('cost: {}'.format(trainer(x,y)))


# In[16]:

w0 = W0.get_value()
_ = imshow(w0[:,0].reshape((28,28)), cmap='gray')


# In[17]:

ERR = T.neq(O2.argmax(axis=-1), Y.argmax(axis=-1))
Ferr = theano.function([X,Y], ERR)
def testnet(x, y):
    testerr = 0.
    for b1,b2 in batch_iterator(x, y, 500):
        testerr += Ferr(b1,b2)
    return testerr.sum()

print('test error: {}, test acc: {}'.format(testnet(test_x, test_y),
       1 - testnet(test_x, test_y) / 10000.))


## Convolutional Nets

# In[19]:

"""
 We can do much better than this with more hidden neurons and dropout.
 Watch Alec Radford's presentation to see how to do that 
 with Python/Theano: https://www.youtube.com/watch?v=S75EdAcXHKk
 For now, lets move on to convnets.
 
"""
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
def conv_rectifier(input_channels, output_channels, filter_dim, X):
    W = sharedX(np.random.normal(0, .001, size=(output_channels,
                                                      input_channels,
                                                      filter_dim,
                                                      filter_dim)))
    b  = sharedX(np.zeros((output_channels,)))
    Z  = conv2d(X,W) + b.dimshuffle('x',0,'x','x')
    DS = max_pool_2d(Z, ds=[2,2])
    O  = T.switch(DS>0, DS, 0)
    return W,b,O

# test
X = T.tensor4('X')
W, b, O = conv_rectifier(1, 9, 5, X)
F = theano.function([X], O)

x = np.ones((5, 1, 28, 28))
print(x.shape)
o = F(x)
o.shape


# In[21]:

Y = T.matrix('Y')
W0, b0, O0 = conv_rectifier(1, 20, 5, X)
W1, b1, O1 = conv_rectifier(20, 50, 5, O0)

# test
F = theano.function([X], O1)
o = F(x)
print(o.shape)


# In[22]:

W2, b2, O2 = rectifier(50*4*4, 500, O1.flatten(2))
W3, b3, O3, cost = softmax(500, 10,  O2, Y)
# Teeeeeest
x = np.ones((128,1,28,28)).astype(floatX)
y = np.ones((128,10)).astype(floatX)
F = theano.function([X, Y], [O3, cost])
z, c = F(x,y)
assert c>0
assert z.shape == (128,10)


# In[23]:

# We need to modify the batch_iterator slightly to serve formated images
def batch_iterator(x, y, batch_size):
    num_batches = x.shape[0] // batch_size
    for i in xrange(0,num_batches):
        # TODO: use random integers instead of consecutive
        #   values to avoid biased gradients
        first = i * batch_size
        last  = (i+1) * batch_size
        x_batch = x[first:last].reshape((batch_size,1,28,28))
        y_pre   = y[first:last]
        y_batch = np.zeros((batch_size, 10))
        for row, col in enumerate(y_pre):
            y_batch[row, col] = 1
        yield (x_batch, y_batch)

for x,y in batch_iterator(train_x, train_y, 10000):
    print('{}, {}'.format(x.shape, y.shape))
print(y[0])
_ = imshow(x[0].reshape((28,28)), cmap='gray')


# In[24]:

params = [W0, b0, W1, b1, W2, b2, W3, b3]
updates = dict()
for p in params:
    updates[p] = p - .01 * T.grad(cost, p)
updates = OrderedDict(updates)
trainer = theano.function([X,Y], cost, updates=updates)


# In[ ]:

num_epochs = 100
for i in range(num_epochs):
    print('-'*10)
    print('Epoch: {}'.format(i))
    for iter,b in enumerate(batch_iterator(train_x, train_y, 128)):
        x = b[0]
        y = b[1]
        last_cost = trainer(x,y)
    print('cost: {}'.format(trainer(x,y)))


# In[ ]:

w0 = W0.get_value()
_ = imshow(w0[0,0,:,:].reshape((5,5)), cmap='gray')


# In[ ]:

ERR = T.neq(O3.argmax(axis=-1), Y.argmax(axis=-1))
Ferr = theano.function([X,Y], ERR)
def testnet(x, y):
    testerr = 0.
    for b1,b2 in batch_iterator(x, y, 500):
        testerr += Ferr(b1,b2)
    return testerr.sum()

print('test error: {}, test acc: {}'.format(testnet(test_x, test_y),
       1 - testnet(test_x, test_y) / 10000.))

