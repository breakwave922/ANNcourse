import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random
import csv

np.random.seed(2000)  # this is what I used to get your random numbers!!!

###feng's code###
# our nonlinear function (and its derivative); lam = 1 (so fixed)
def acti(x, derive=False):
    if derive:
        return 1 - x*x #(np.square(x))
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# read training/test images
train1_fname='./Part2_dat/Part2_1_Train.csv'
train3_fname='./Part2_dat/Part2_3_Train.csv'
'''with open('./Part2_dat/Part2_1_Train.csv') as f:
    reader = csv.reader(f)
    for col in reader:
        print(col)'''
train1_dat = np.genfromtxt(train1_fname, delimiter=',')
train3_dat = np.genfromtxt(train3_fname, delimiter=',')

###last column is the right value that img represents
train1_dat=np.append(train1_dat,np.ones((train1_dat.shape[0],1)),axis=1)
train3_dat=np.append(train3_dat,3*np.ones((train3_dat.shape[0],1)),axis=1)
train13_dat=np.concatenate((train1_dat,train3_dat),axis=0)  ##merge two data

#define several dimension para
img_dim=28   ##image dim
layerh_n=2   ## #of neurons in hidden layer
layero_n=2
#imagesc( reshape(Matrix1_Test(:,1),[28 28]) )

#initialize wgt
n1_w = np.random.normal(0, 1, (img_dim,img_dim))   ##hidden layer feature maps-1
n2_w = np.random.normal(0, 1, (img_dim,img_dim))   ##hidden layer feature maps-2
b1 = np.random.normal(0, 1, (1,)) #bias
b2 = np.random.normal(0, 1, (1,)) #bias

no_w1 = np.random.normal(0, 1, (layerh_n+1,))    ## for output layer neuron1,including bias
no_w2 = np.random.normal(0, 1, (layerh_n+1,))    ## for output layer neuron2,including bias


# learning rate
eta = 0.2

# target output
y=np.array([[1,0],[0,1]])  ##first/sec row corrsponds to class1&2

###############################################
# Epochs
###############################################
epoch = 10  # how many epochs? (each epoch will pass all 4 data points through)
err = np.zeros((epoch, 1))  # lets record error to plot (get a convergence plot)
inds = np.arange(np.size(train13_dat,0))  # array of our training indices (data point index references)

for k in range(epoch):
    # init error
    err[k] = 0

    # random shuffle of data each epoch
    # inds = np.random.permutation(inds)
    for i in range(np.size(inds)):
        # what index?
        inx = inds[i]
    # forward pass
        v = np.ones(layerh_n+1,)  # last one is for bias
        for j in range(layerh_n):
            if j==0:
                v[j] = np.multiply(train13_dat[inx,0:-1].reshape(img_dim,img_dim),n1_w).sum()+b1
                v[j] = acti(v[j])
            else:
                v[j] = np.multiply(train13_dat[inx,0:-1].reshape(img_dim,img_dim),n2_w).sum()+b2
                v[j] = acti(v[j])

        oo = np.array([np.dot(v, no_w1),np.dot(v, no_w2)])  # output neuron 0&1 fires, taking hidden neuron 1 and 2 as input
        o = acti(oo)  # result of output 0&1 !!!


         ###calculating error
        if train13_dat[inx,-1]==1:
            err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[0,:]), 2.0)).sum()
        else:
            err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[1,:]), 2.0)).sum()

        # backprop time!!! ################################

        # output layer
        delta_ow1 = np.ones((layero_n + 1, 1))   ##last one is for bias
        delta_ow2 = np.ones((layero_n + 1, 1))
        delta_1 = o - y[inx]
        delta_2 = acti(o, derive=True)
        delta_ow1 = v * delta_1[0] * delta_2[0]
        delta_ow2 = v * delta_1[1] * delta_2[1]

