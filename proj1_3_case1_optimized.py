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

np.random.seed(1000)  # this is what I used to get your random numbers!!!

###feng's code###
# our nonlinear function (and its derivative); lam = 1 (so fixed)
'''def acti(x, derive=False):
    if derive:
        return 1 - x*x #(np.square(x))
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))'''

def acti(x, lamda=0.3, derive=False, entropy=False):
    if derive:
        return lamda * (x * (1 - x))
    if entropy:
        return lamda * 1.0 + (x - x)
    return 1 / (1 + np.exp(-lamda*x))


'''def slice2D(m,start,win_d): #### m-matrix, start-window upper left corner cordin., win_d:window dimension
    return (m[start[0]:start[0]+win_d,start[1]:start[1]+win_d])'''

def convimg(v,m,img_dim,share_wgt_dim,wgt,b,f):     ### only applicable to no-padding, strike=1 case
    for jup in range(img_dim-share_wgt_dim+1):  # scan from up to down
        for jlr in range(img_dim-share_wgt_dim+1):  # scan from left to right
                v[f,jup,jlr] = np.multiply(m[jup:jup+share_wgt_dim,jlr:jlr+share_wgt_dim], wgt).sum() + b
    return

def sharewgt_imginput(imginput,m,share_wgt_dim,sliding_o):
    for jup in range(share_wgt_dim):  # scan from up to down
        for jlr in range(share_wgt_dim):  # scan from left to right
            imginput[jup,jlr,:,:] = (m[jup:jup+sliding_o,jlr:jlr+sliding_o])

#define several dimension para
img_dim=28   ##image dim
share_wgt_dim=7 ## share wgt dim
feature_n=16    ## feature map number
sliding_o=img_dim-share_wgt_dim+1  ## sliding output size
layerh_n=sliding_o**2*feature_n   ## #of neurons in hidden layer
#layerh_n2=128   ###second hidden layer
layero_n=10   ## # of neurons in output layer


'''pixesdiv=np.zeros((share_wgt_dim,share_wgt_dim,sliding_o,sliding_o))
for jup in range(share_wgt_dim):
    for jlr in range(share_wgt_dim):
        #pixesdiv[jup, jlr,:,:]=np.meshgrid(range(jup,jup+sliding_o),range(jlr,jlr+sliding_o))
        x,y=np.ogrid[jup:jup+sliding_o,jlr:jlr+sliding_o]
        pixesdiv[jup,jlr]=x+y'''

'''coordinates = np.zeros((share_wgt_dim,share_wgt_dim,sliding_o,sliding_o))
for jup in range(share_wgt_dim):
    for jlr in range(share_wgt_dim):
        for x in range(sliding_o):
            for y in range(sliding_o):
                coordinates[jup,jlr,x,y]=np.array([x+jup,y+jlr])'''


# read training/test images
train_name=[]
train_dat=np.zeros(shape=(10,500,img_dim**2+1))  ##last column is for values
for i in range(10):
    train_dat_temp=np.genfromtxt('./Part3_dat/Part3_'+str(i)+'_Train.csv', delimiter=',')
    train_dat[i, :, :]=np.append(train_dat_temp,i*np.ones((train_dat.shape[1],1)),axis=1)

#initialize wgt
wgt_width=0.1
nh_w = np.random.normal(0, wgt_width, (feature_n,share_wgt_dim,share_wgt_dim))/np.sqrt(feature_n*share_wgt_dim*share_wgt_dim)   ##hidden layer feature maps
#n2_w = np.random.normal(0, 1, (img_dim,img_dim))   ##hidden layer feature maps-2
b_h = np.random.normal(0, wgt_width, (feature_n,)) #bias
#b_h2 = np.random.normal(0, 1, (1,)) #bias

## from first hidden layer to second hidden layer
#nh2_w = np.random.normal(0, wgt_width, (layerh_n2,feature_n,sliding_o,sliding_o))    ## for first hidden layer to second hidden layer
#b_h2 = np.random.normal(0, wgt_width, (layerh_n2,))

#no_w = np.random.normal(0, 1, (layero_n,layerh_n+1))    ## for output layer neuron1,including bias, each row of wgt is used for 1 output neuron
#no_w = np.random.normal(0, wgt_width, (layero_n,layerh_n2+1))    ## for output layer neuron1, last one is bias
no_w = np.random.normal(0, wgt_width, (layero_n,feature_n,sliding_o,sliding_o))/np.sqrt(layero_n*feature_n*sliding_o**2)     ## for output layer neuron1,
b_o = np.random.normal(0, wgt_width, (layero_n,)) #bias--output layer

# learning rate
eta = 3

# target output
y=np.diag(np.ones((layero_n,)))
y[0]=0


###############################################
# Epochs
###############################################
traindata=train_dat
epoch =50 # how many epochs?
err = np.zeros((epoch, 1))  # lets record error to plot (get a convergence plot)
inds = np.arange(np.size(traindata,0))  # array of our training indices (data point index references)
inds_dig=np.arange(np.size(traindata,1))
img_dig=10  ###for choosing how many img per dig to train
#inds=np.arange(1)
#inds[0]=4
#f = IntProgress(min=0, max=epoch)  # instantiate the bar (you can "see" how long alg takes to run)
#display(f)  # display the bar!

for k in range(epoch):
    print("epoch:",k)
    # init error
    err[k] = 0

    # random shuffle of data each epoch
    inds = np.random.permutation(inds)

    for i in range(np.size(inds)):
        inds_dig = np.random.permutation(inds_dig)
        for imim in range(img_dig):
            #print("data:", i)
            # what index?
            inx = inds[i]
            img_data=traindata[inx,inds_dig[imim],0:-1].reshape(img_dim,img_dim)  ### convert to img matrix
        # forward pass
            v = np.ones((feature_n,sliding_o,sliding_o))
            for j in range(feature_n):
                convimg(v,m=img_data,img_dim=img_dim,share_wgt_dim=share_wgt_dim,wgt=nh_w[j,:,:],b=b_h[j],f=j)
            v = acti(v)

            oo = np.zeros((layero_n,))
            # o=np.zeros((layero_n,))
            for kk in range(layero_n):
                oo[kk] = np.multiply(no_w[kk, :, :, :], v[:, :, :]).sum() + b_o[kk]
            o = acti(oo)  # result of output 0&1 !!!

            ###calculating error
            #err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[inx,:]), 2.0)).sum()
            err[k]=err[k]+((1-y[inx,:])*np.log(1-o)-y[inx,:]*np.log(o)).sum()

            # backprop time!!! ################################
            # output layer
            delta_ow = np.zeros((layero_n, feature_n, sliding_o, sliding_o))
            delta_ob = np.zeros((layero_n,))
            delta_1 = o - y[inx, :]

            delta_2 = acti(o, entropy=True)
           # print('delta2', delta_2)


            delta_hw=np.zeros((layero_n,feature_n, sliding_o, sliding_o))   ###for calculating hidden layer wgt updates

            for uuu in range(layero_n):
                delta_ow[uuu, :, :, :] = delta_1[uuu] * delta_2[uuu] * v[:, :, :]
                delta_hw[uuu, :, :, :] = delta_1[uuu] * delta_2[uuu] * no_w[uuu,:, :, :]

                #delta_ow=np.reshape(np.array([(delta_1 * delta_2)]).T @ np.array([v.flatten()]),(layero_n,feature_n,sliding_o,sliding_o))


            delta_ob = delta_1 * delta_2

            np.array([(delta_1 * delta_2)]).T            #### for hidden layer
            delta_hw=delta_hw.sum(axis=0)   ###sum up for calculating hidden layer

            delta_3 = acti(v[:, :, :], derive=True)

            delta_hw1 = delta_hw * delta_3
            delta_bh=(delta_hw1.sum(axis=2)).sum(axis=1)   ###for share-bias update

            imginput=np.zeros((share_wgt_dim,share_wgt_dim,sliding_o,sliding_o))
            sharewgt_imginput(imginput,m=img_data,share_wgt_dim=share_wgt_dim,sliding_o=sliding_o)

            delta_nh=np.zeros((feature_n, share_wgt_dim, share_wgt_dim, sliding_o, sliding_o))
            for fff in range(feature_n):
                delta_nh[fff,:,:,:,:]=delta_hw1[fff,:,:]*imginput ###imgpixvalue in ranges imag(updownleftright) (feature_n,share_wgt_dim,share_wgt_dim)
            delta_nh=(delta_nh.sum(axis=4)).sum(axis=3)    ### sum up on the last two dimeison, to have delta wgt for share wgt

            '''delta_nh = np.zeros((feature_n, share_wgt_dim, share_wgt_dim))
for fff in range (feature_n):
    for jup in range(share_wgt_dim):
        for jlr in range(share_wgt_dim):
            delta_nh[fff, jup, jlr]  = (delta_hw1[fff, :, :] * imginput[jup,jlr,:,:]).sum()'''

            # update rule, so old value + eta weighted version of delta's above!

            no_w = no_w + (-1.0) * eta * delta_ow

            nh_w = nh_w + (-1.0) * eta * delta_nh
            b_h = b_h + (-1.0) * eta * delta_bh

        #print('err',err)

# plot it
plt.plot(err)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('Convergence Plot')
plt.show()
print(err)

# run through out data, what should label be, what did we get?
inds = np.random.permutation(inds)
# random shuffle of data each epoch
    # inds = np.random.permutation(inds)
for i in range(np.size(inds)):
    # print("data:", i)
    # what index?
    inx = inds[i]
    img_data = traindata[inx, 80, 0:-1].reshape(img_dim, img_dim)  ### convert to img matrix
    # forward pass
    v = np.ones((feature_n, sliding_o, sliding_o))
    for j in range(feature_n):
        convimg(v, m=img_data, img_dim=img_dim, share_wgt_dim=share_wgt_dim, wgt=nh_w[j, :, :], b=b_h[j], f=j)
    v = acti(v)

    vh2 = np.ones((layerh_n2 + 1,))  ###including bias

    ###from first hidden to sec hidden
    for sss in range(layerh_n2):
        vh2[sss] = np.dot(nh2_w[sss, :, :, :].flatten(), v.flatten()) + b_h2[
            sss]  ### or np.multiply(nh2_w[sss,:,:], v).sum()+ b_h2[sss]
        vh2[sss] = acti(vh2[sss])

    ###output layer
    oo = no_w @ vh2
    o = acti(oo)  # result of output 0&1 !!!
    print("Sample " + str(i) + ": label " + str(traindata[inx,80,-1]) + ": got " + str(o))


#### using final trained wgt to test
'''testdata=train13_dat
inds = np.arange(np.size(testdata,0))  # array of our training indices (data point index references)
inds = np.random.permutation(inds)
for i in range(np.size(inds)):
        # what index?
      inx = inds[i]
      img_data = traindata[inx, 0:-1].reshape(img_dim, img_dim)  ### convert to img matrix
        # forward pass
      v = np.ones((feature_n, sliding_o, sliding_o))
      for j in range(feature_n):
            convimg(v, m=img_data, img_dim=img_dim, share_wgt_dim=share_wgt_dim, wgt=nh_w[j, :, :], b=b_h[j], f=j)
      v = acti(v)
        # oo = np.array([np.dot(v.T, no_w),np.dot(v, no_w2)])  # output neuron 0&1 fires, taking hidden neuron 1 and 2 as input
        # vv = np.append(v.flatten(), 1)    ## append for bias
        # oo = no_w @ vv
      oo = np.zeros((layero_n,))
        # o=np.zeros((layero_n,))
      for kk in range(layero_n):
          oo[kk] = np.multiply(no_w[kk, :, :, :], v[:, :, :]).sum() + b_o[kk]
      o = acti(oo)  # result of output 0&1 !!!'''



