import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
from ipywidgets import IntProgress
import sys
import random
import csv
import time

np.random.seed(2000)  # this is what I used to get your random numbers!!!

###feng's code###
# our nonlinear function (and its derivative); lam = 1 (so fixed)
def acti(x, derive=False):
    if derive:
        return 1 - x*x #(np.square(x))
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def slice2D(m,start,win_d): #### m-matrix, start-window upper left corner cordin., win_d:window dimension
    return (m[start[0]:start[0]+win_d,start[1]:start[1]+win_d])

def convimg(v,m,img_dim,share_wgt_dim,wgt,b,f):     ### only applicable to no-padding, strike=1 case
    for jup in range(img_dim-share_wgt_dim+1):  # scan from up to down
        for jlr in range(img_dim-share_wgt_dim+1):  # scan from left to right
                v[f,jup,jlr] = np.multiply(m[jup:jup+share_wgt_dim,jlr:jlr+share_wgt_dim], wgt).sum() + b
                #vv[kk] = acti(vv[k])
                #kk += 1
    return




# read training/test images

train1_fname='./Part2_dat/Part2_1_Train.csv'
train3_fname='./Part2_dat/Part2_3_Train.csv'

train1_dat = np.genfromtxt(train1_fname, delimiter=',')
train3_dat = np.genfromtxt(train3_fname, delimiter=',')

test1_fname='./Part2_dat/Part2_1_Test.csv'
test3_fname='./Part2_dat/Part2_3_Test.csv'

test1_dat = np.genfromtxt(test1_fname, delimiter=',')
test3_dat = np.genfromtxt(test3_fname, delimiter=',')

###last column is the right value that img represents
train1_dat=np.append(train1_dat,np.ones((train1_dat.shape[0],1)),axis=1)
train3_dat=np.append(train3_dat,3*np.ones((train3_dat.shape[0],1)),axis=1)
train13_dat=np.concatenate((train1_dat,train3_dat),axis=0)  ##merge two data

test1_dat=np.append(test1_dat,np.ones((test1_dat.shape[0],1)),axis=1)
test3_dat=np.append(test3_dat,3*np.ones((test3_dat.shape[0],1)),axis=1)
test13_dat=np.concatenate((test1_dat,test3_dat),axis=0)  ##merge two data



#define several dimension para
img_dim=28   ##image dim
share_wgt_dim=7 ## share wgt dim
feature_n=16    ## feature map number
sliding_o=img_dim-share_wgt_dim+1  ## sliding output size
layerh_n=sliding_o**2*feature_n   ## #of neurons in hidden layer
layero_n=10   ## # of neurons in output layer
#imagesc( reshape(Matrix1_Test(:,1),[28 28]) )
#plt.imshow(train13_dat[0,0:-1].reshape(img_dim,img_dim))

#initialize wgt
nh_w = np.random.normal(0, 1, (feature_n,share_wgt_dim,share_wgt_dim))   ##hidden layer feature maps
#n2_w = np.random.normal(0, 1, (img_dim,img_dim))   ##hidden layer feature maps-2
b_h = np.random.normal(0, 1, (feature_n,)) #bias
#b_h2 = np.random.normal(0, 1, (1,)) #bias

no_w = np.random.normal(0, 1, (layero_n,layerh_n+1))    ## for output layer neuron1,including bias, each row of wgt is used for 1 output neuron


# learning rate
eta = 0.1

# target output
y=np.diag(np.ones((layero_n,)))


###############################################
# Epochs
###############################################
traindata=train13_dat
epoch = 50 # how many epochs?
err = np.zeros((epoch, 1))  # lets record error to plot (get a convergence plot)
inds = np.arange(np.size(traindata,0))  # array of our training indices (data point index references)
f = IntProgress(min=0, max=epoch)  # instantiate the bar (you can "see" how long alg takes to run)
display(f)  # display the bar!

for k in range(epoch):
    # init error
    err[k] = 0

    # random shuffle of data each epoch
    inds = np.random.permutation(inds)
    for i in range(np.size(inds)):
        # what index?
        inx = inds[i]
        img_data=traindata[inx,0:-1].reshape(img_dim,img_dim)  ### convert to img matrix
    # forward pass
        v = np.ones((feature_n,sliding_o,sliding_o))
        for j in range(feature_n):
            convimg(v,m=img_data,img_dim=img_dim,share_wgt_dim=share_wgt_dim,wgt=nh_w[j,:,:],b=b_h[j],f=j)

        #oo = np.array([np.dot(v.T, no_w),np.dot(v, no_w2)])  # output neuron 0&1 fires, taking hidden neuron 1 and 2 as input
        vv = np.append(v.flatten(), 1)    ## append for bias
        oo = no_w @ vv
        o = acti(oo)  # result of output 0&1 !!!


         ###calculating error
        if traindata[inx,-1]<3.0:
            err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[1,:]), 2.0)).sum()
        else:
            err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[3,:]), 2.0)).sum()

        # backprop time!!! ################################
        # output layer
        delta_ow = np.zeros((layero_n, layerh_n+1))   ##last col is for bias

        if traindata[inx, -1] <3.0:
            delta_1 = o - y[1, :]
        else:
            delta_1 = o - y[3, :]
        delta_2 = acti(o, derive=True)

        for uuu in range(layero_n):
            for hhh in range(layerh_n+1):
                delta_ow[uuu,hhh]= delta_1[uuu]*delta_2[uuu]*v[hhh]

         # for hidden layer
        delta_nh=np.zeros((feature_n,share_wgt_dim,share_wgt_dim))
        delta_bh=np.zeros(layerh_n,)
        for hh in range(feature_n):
            delta_h = acti(v[hh, :, :], derive=True)
            for mm in range(layero_n):
                for hhh in range(sliding_o**2):
                    delta_1[mm] * delta_2[mm] * no_w[mm, hh*sliding_o+hhh]*delta_h[hh,hhh]*img_data[jup:jup+share_wgt_dim,jlr:jlr+share_wgt_dim]

            for cc in range(sliding_o**2):  #per feature
                for mm in range(layero_n):
                    delta_nh[hh,:,:]+=delta_1[mm] * delta_2[mm] * no_w[mm,hh] * delta_h*traindata[inx,0:-1].reshape(img_dim,img_dim)
                    delta_bh[hh]+=delta_1[mm] * delta_2[mm] * no_w[mm,hh]* delta_h

        # update rule, so old value + eta weighted version of delta's above!
        nh_w = nh_w + (-1.0) * eta * delta_nh
        b_h = b_h + (-1.0) * eta * delta_bh
        no_w = no_w + (-1.0) * eta * delta_ow

# plot it
plt.plot(err)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('Convergence Plot')
plt.show()

# run through out data, what should label be, what did we get?
inds = np.random.permutation(inds)
for i in range(np.size(inds)):
        # what index?
    inx = inds[i]
    # forward pass
    v = np.ones(layerh_n+1,)  # last one is for bias
    for j in range(layerh_n):
        v[j] = np.multiply(traindata[inx,0:-1].reshape(img_dim,img_dim),nh_w[:,:,j]).sum()+b_h[j]
        v[j] = acti(v[j])

        #oo = np.array([np.dot(v.T, no_w),np.dot(v, no_w2)])  # output neuron 0&1 fires, taking hidden neuron 1 and 2 as input
    oo = no_w @ v
    o = acti(oo)  # result of output 0&1 !!!


    print("Sample " + str(i) + ": label " + str(traindata[inx,-1]) + ": got " + str(o))


#### using final trained wgt to test
testdata=test13_dat
inds = np.arange(np.size(testdata,0))  # array of our training indices (data point index references)
inds = np.random.permutation(inds)
for i in range(np.size(inds)):
        # what index?
    inx = inds[i]
    # forward pass
    v = np.ones(layerh_n+1,)  # last one is for bias
    for j in range(layerh_n):
        v[j] = np.multiply(testdata[inx,0:-1].reshape(img_dim,img_dim),nh_w[j,:,:]).sum()+b_h[j]
        v[j] = acti(v[j])

        #oo = np.array([np.dot(v.T, no_w),np.dot(v, no_w2)])  # output neuron 0&1 fires, taking hidden neuron 1 and 2 as input
    oo = no_w @ v
    o = acti(oo)  # result of output 0&1 !!!


    print("Sample " + str(i) + ": label " + str(testdata[inx,-1]) + ": got " + str(o))
    if testdata[inx,-1]<3.0:
        colors='red'
        l ='1'
        target1=plt.scatter(o[0], o[1], c=colors, alpha=0.5)
    else:
        colors = 'blue'
        l = '3'
        target3=plt.scatter(o[0], o[1], c=colors, alpha=0.5)

plt.legend((target1, target3),
           ('target1', 'target3'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)
xvec = np.linspace(-2.,2.,100)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.legend()
plt.xlim(-2,  2)
plt.ylim(-2, 2)

plt.annotate('1', xy=(1, 0), xytext=(1.5, 0.5),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
plt.annotate('3', xy=(0, 1), xytext=(0.5, 1.5),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )

plt.show()


