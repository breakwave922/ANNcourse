import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random
np.random.seed(2000)  # this is what I used to get your random numbers!!!

###feng's code###
# our nonlinear function (and its derivative); lam = 1 (so fixed)
def tanh(x, derive=False):
    if derive:
        return 1 - x*x #(np.square(x))
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# define six training samples
X = np.array ([[0, 0, 0.8, 0.4, 0.4, 0.1, 0, 0, 0, 1], # data point (x,y) = (1,1), with homogenization 1 for bias
 [0, 0.3, 0.3, 0.8, 0.3, 0, 0, 0, 0, 1], # data point (1,0)
 [0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 0, 1],
 [0, 0, 0, 0, 0, 0.8, 0.4, 0.4, 0.1, 1],
 [0.8, 0.4, 0.4, 0.1, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 1],])

# its labels
y = np.array([[-1],  # class 1
              [1],  # class 1
              [1],
              [-1],
              [-1],
              [1],
              ])

# learning rate
eta = 0.2

# initialize weights
'''n1_w = np.array([[1.73673761, 1.89791391, - 2.10677342, - 0.14891209, 0.58306155]])
n1_w = np.transpose(n1_w)
n2_w = np.array([[-2.25923303, 0.13723954, -0.70121322, -0.62078008, -0.47961976]])
n2_w = np.transpose(n2_w)

n3_w = np.array([[1.20973877, -1.07518386, 0.80691921, -0.29078347, -0.22094764, -0.16915604,
                  1.10083444, 0.08251052, -0.00437558, -1.72255825, 1.05755642, -2.51791281,
                  -1.91064012]])
n3_w = np.transpose(n3_w)'''

n1_w = np.random.normal(0, 1, (5,))
n2_w = np.random.normal(0, 1, (5,))
n3_w = np.random.normal(0, 1, (13,))

share_wgt_size=4    #define share share wgt size
layer2_n=6 # define #of neurons for layer 2 for each color

###############################################
# Epochs
###############################################
epoch = 1  # how many epochs? (each epoch will pass all 4 data points through)
err = np.zeros((epoch, 1))  # lets record error to plot (get a convergence plot)
inds = np.arange(np.size(X,0))  # array of our training indices (data point index references)

for k in range(epoch):
    # init error
    err[k] = 0

    # random shuffle of data each epoch
    # inds = np.random.permutation(inds)
    for i in range(np.size(inds)):
        # what index?
        inx = inds[i]

        # forward pass
        v = np.ones((2*layer2_n+1,))   # last one is for bias
        for j in range(2*layer2_n):
            if j<layer2_n: ###for green color
               v[j] = np.dot(X[inx, j:j+share_wgt_size], n1_w[0:share_wgt_size])+n1_w[-1]*X[inx, -1] # neuron 1 fires (x as input)
               v[j] = tanh(v[j])
            else:  ###for orange color
               v[j] = np.dot(X[inx, j-layer2_n:j-layer2_n + share_wgt_size], n2_w[0:share_wgt_size ]) + n2_w[-1]*X[inx, -1]  # neuron 1 fires (x as input)
               v[j] = tanh(v[j])  # neuron j sigmoid

        oo = np.dot(np.transpose(v), n3_w[:])  # neuron 3 fires, taking neuron 1 and 2 as input
        o = tanh(oo)  # hey, result of our net!!!
        #print(o)
        # error
        err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[inx]), 2.0))
        #print(err)
        # backprop time!!!

        # output layer
        delta_1 = o - y[inx]
        delta_2 = tanh(o, derive=True)

        # now, lets prop it back to the weights and bias
        delta_ow = np.ones((2*layer2_n+1, ))   ###for output wgt

        delta_hw1 = np.ones((share_wgt_size+1, layer2_n))  ####for hidden layer wgt update
        delta_hw2 = np.ones((share_wgt_size+1, layer2_n))  ####for hidden layer wgt update

        for j in range(2 * layer2_n+1):
            delta_ow[j] = v[j] * (delta_1 * delta_2)  ##including bias
        #delta_ow[-1]=delta_1 * delta_2   ###for bias

        # for hidden layer

        for ii in range(share_wgt_size+1):    ###for weight updata-first set
            for j in range(layer2_n):
                delta_3 = tanh(v[j], derive=True)
                if ii < share_wgt_size:  ###for wgt
                   delta_hw1[ii,j] = delta_1 * delta_2 * n3_w[j] * delta_3 * X[inx, ii + j]   #for green layer
                else:
                   delta_hw1[ii,j] = delta_1 * delta_2 * n3_w[j] * delta_3    #for green layer-bias

        for ii in range(share_wgt_size+1):    ###for weight updata-2nd set
            for j in range(layer2_n):
                delta_3 = tanh(v[j+layer2_n], derive=True)
                if ii < share_wgt_size:  ###for wgt
                   delta_hw2[ii,j]= delta_1 * delta_2 * n3_w[j+layer2_n] * delta_3 * X[inx,ii+j]   #for orange layer
                else:      ###for bias
                   delta_hw2[ii,j] = delta_1 * delta_2 * n3_w[j+layer2_n] * delta_3  # for orange layer-bias

        delta_hw1_sum = np.sum(delta_hw1,1)
        delta_hw2_sum = np.sum(delta_hw2,1)
        #delta_hw1_sum=np.transpose(delta_hw1_sum)
        #delta_hw2_sum=np.transpose(delta_hw2_sum)

        '''
          or delta_hw1_sum = np.sum(delta_hw1,1)
          delta_hw1_sum = np.transpose([delta_hw1_sum])
          delta_hw2_sum = np.sum(delta_hw2,1)
          delta_hw2_sum = np.transpose([delta_hw2_sum])
        '''
 # update rule, so old value + eta weighted version of delta's above!!!
        n1_w = n1_w - eta * delta_hw1_sum
        n2_w = n2_w - eta * delta_hw2_sum
        n3_w = n3_w - eta * delta_ow
    print(n1_w)
    print(n2_w)
    print(n3_w)

# plot it
plt.plot(err)
plt.ylabel('some numbers')
plt.show()

# what were the values (just do forward pass)
for i in range(np.size(X,0)):
# what index?
    inx = inds[i]

# forward pass
    v = np.ones((2 * layer2_n, 1))
    for j in range(2 * layer2_n):
        if j < layer2_n:  ###for green color
           v[j] = np.dot(X[inx, j:j + share_wgt_size], n1_w[0:share_wgt_size]) + n1_w[
         -1]*X[inx, -1]  # neuron 1 fires (x as input)
           v[j] = tanh(v[j])
    else:  ###for orange color
           v[j] = np.dot(X[inx, j - layer2_n:j - layer2_n + share_wgt_size], n2_w[0:share_wgt_size]) + n2_w[
         -1]*X[inx, -1]  # neuron 1 fires (x as input)
           v[j] = tanh(v[j])  # neuron j sigmoid

    oo = np.dot(np.transpose(v), n3_w[:-1])  # neuron 3 fires, taking neuron 1 and 2 as input
    o = tanh(oo)  # hey, result of our net!!!
    print(str(i) + ": produced: " + str(o) + " wanted " + str(y[inx]))