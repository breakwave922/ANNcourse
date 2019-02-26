# libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random
from ipywidgets import IntProgress
import time

# parameters
np.random.seed(2000)  # this is what I used to get your random numbers!!!
eta = 0.2  # learning rate
epoch = 10  # how many epochs to run?
RandomShuffle = 0  # do we want to randomly shuffle our data?


# our nonlinear function (and its derivative)
def tanh(x, derive=False):
    if derive:
        return 1 - x * x
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))


# our data set
X = np.array([
    [0, 0, 0.8, 0.4, 0.4, 0.1, 0, 0, 0],
    [0, 0.3, 0.3, 0.8, 0.3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 0],
    [0, 0, 0, 0, 0, 0.8, 0.4, 0.4, 0.1],
    [0.8, 0.4, 0.4, 0.1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3],
])
# its labels
y = np.array([[-1],  # class 1
              [1],  # class 2
              [1],
              [-1],
              [-1],
              [1]
              ])

# initialize weights with random numbers
# again, these are the numbers I put in your assignment PDF
n1_w = np.random.normal(0, 1, (5,))
n2_w = np.random.normal(0, 1, (5,))
n3_w = np.random.normal(0, 1, (13,))

# print out the values when we started!!!
print("Started with")
print("shared weight 1: " + str(n1_w))
print("shared weight 2: " + str(n2_w))
print("net weights: " + str(n3_w))

###############################################
# Epochs
###############################################
err = np.zeros((epoch, 1))  # lets record error so we can make a convergence plot
inds = np.asarray([0, 1, 2, 3, 4, 5])  # easy (to read) way to deal with shuffle data or not (indices)
f = IntProgress(min=0, max=epoch)  # instantiate the bar (you can "see" how long alg takes to run)
display(f)  # display the bar!
for k in range(epoch):  # our epoch iterator

    f.value += 1  # increment the progress bar

    # init error (will calc below)
    err[k] = 0

    # random shuffle of data each epoch?
    if (RandomShuffle == 1):
        inds = np.random.permutation(inds)

    # go through each sample (so online learning vs. batch or minibatch)
    for i in range(6):

        # what index will we look at in this loop?
        inx = inds[i]

        # forward pass #####################################

        # first layer
        s = np.zeros((12,))
        for j in range(6):  # first shared weight group
            s[j] = np.dot(X[inx, j:(j + 4)], n1_w[0:4]) + n1_w[4] * 1  # last part is our bias!
            s[j] = tanh(s[j])
        for j in range(6):  # second shared weight group
            s[j + 6] = np.dot(X[inx, j:(j + 4)], n2_w[0:4]) + n2_w[4] * 1
            s[j + 6] = tanh(s[j + 6])
        # second layer
        o = np.dot(np.transpose(s), n3_w[0:12]) + n3_w[12] * 1
        oo = tanh(o)

        # error
        err[k] = err[k] + ((1.0 / 2.0) * np.power((oo - y[inx]), 2.0))

        # backprop time!!! ################################

        # output layer
        delta_1 = (-1.0) * (y[inx] - oo)
        delta_2 = tanh(oo, derive=True)
        delta_w2 = np.ones((13,))
        for j in range(12):
            delta_w2[j] = s[j] * (delta_1 * delta_2)
        delta_w2[12] = 1 * (delta_1 * delta_2)

        # first shared weight's
        delta_sigs = np.ones((6,))
        for j in range(6):
            delta_sigs[j] = tanh(s[j], derive=True)
        delta_w11 = np.zeros((5,))
        for j in range(4):
            for m in range(6):
                delta_w11[j] = delta_w11[j] + (X[inx, j + m] * delta_sigs[m] * ((delta_1 * delta_2) * n3_w[m]))
        for m in range(6):
            delta_w11[4] = delta_w11[4] + (1 * delta_sigs[m] * ((delta_1 * delta_2) * n3_w[m]))

            # second shared weight's
        delta_sigs = np.ones((6,))
        for j in range(6):
            delta_sigs[j] = tanh(s[j + 6], derive=True)
        delta_w12 = np.zeros((5,))
        for j in range(4):
            for m in range(6):
                delta_w12[j] = delta_w12[j] + (X[inx, j + m] * delta_sigs[m] * ((delta_1 * delta_2) * n3_w[m + 6]))
        for m in range(6):
            delta_w12[4] = delta_w12[4] + (1 * delta_sigs[m] * ((delta_1 * delta_2) * n3_w[m + 6]))

            # update rule, so old value + eta weighted version of delta's above!
        n1_w = n1_w + (-1.0) * eta * delta_w11
        n2_w = n2_w + (-1.0) * eta * delta_w12
        n3_w = n3_w + (-1.0) * eta * delta_w2

# print out what we found

print("Done")
print("shared weight 1: " + str(n1_w))
print("shared weight 2: " + str(n2_w))
print("net weights: " + str(n3_w))

# plot it
plt.plot(err)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('Convergence Plot')
plt.show()

# last, run through out data, what should label be, what did we get?
inds = np.asarray([0, 1, 2, 3, 4, 5])
for i in range(6):

    inx = inds[i]

    s = np.zeros((12,))
    for j in range(6):
        s[j] = np.dot(X[inx, j:(j + 4)], n1_w[0:4]) + n1_w[4] * 1
        s[j] = tanh(s[j])
    for j in range(6):
        s[j + 6] = np.dot(X[inx, j:(j + 4)], n2_w[0:4]) + n2_w[4] * 1
        s[j + 6] = tanh(s[j + 6])

    o = np.dot(np.transpose(s), n3_w[0:12]) + n3_w[12] * 1
    oo = tanh(o)

    print("Sample " + str(i) + ": label " + str(y[inx]) + ": got " + str(oo))