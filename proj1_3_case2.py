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

np.random.seed(3000)  # this is what I used to get your random numbers!!!

###feng's code###
# our nonlinear function (and its derivative); lam = 1 (so fixed)
'''def acti(x, derive=False):
    if derive:
        return 1 - x*x #(np.square(x))
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))'''

def acti(x, lamda=0.05, derive=False):
    if derive:
        return lamda * (x * (1 - x))
    return 1 / (1 + np.exp(-lamda*x))


'''def slice2D(m,start,win_d): #### m-matrix, start-window upper left corner cordin., win_d:window dimension
    return (m[start[0]:start[0]+win_d,start[1]:start[1]+win_d])'''

def convimg(v,m,img_dim,share_wgt_dim,wgt,b,f):     ### only applicable to no-padding, strike=1 case
    for jup in range(img_dim-share_wgt_dim+1):  # scan from up to down
        for jlr in range(img_dim-share_wgt_dim+1):  # scan from left to right
                v[f,jup,jlr] = np.multiply(m[jup:jup+share_wgt_dim,jlr:jlr+share_wgt_dim], wgt).sum() + b
                #vv[kk] = acti(vv[k])
                #kk += 1
    return

#define several dimension para
img_dim=28   ##image dim
share_wgt_dim=7 ## share wgt dim
feature_n=16    ## feature map number
sliding_o=img_dim-share_wgt_dim+1  ## sliding output size
layerh_n=sliding_o**2*feature_n   ## #of neurons in hidden layer
layerh_n2=128   ###second hidden layer
layero_n=10   ## # of neurons in output layer


# read training/test images
train_name=[]
train_dat=np.zeros(shape=(10,500,img_dim**2+1))  ##last column is for values
for i in range(10):
    train_dat_temp=np.genfromtxt('./Part3_dat/Part3_'+str(i)+'_Train.csv', delimiter=',')
    train_dat[i, :, :]=np.append(train_dat_temp,i*np.ones((train_dat.shape[1],1)),axis=1)

#initialize wgt
wgt_width=0.05
nh_w = np.random.normal(0, wgt_width, (feature_n,share_wgt_dim,share_wgt_dim))   ##hidden layer feature maps
#n2_w = np.random.normal(0, 1, (img_dim,img_dim))   ##hidden layer feature maps-2
b_h = np.random.normal(0, wgt_width, (feature_n,)) #bias
#b_h2 = np.random.normal(0, 1, (1,)) #bias

## from first hidden layer to second hidden layer
nh2_w = np.random.normal(0, wgt_width, (layerh_n2,feature_n,sliding_o,sliding_o))    ## for first hidden layer to second hidden layer
b_h2 = np.random.normal(0, wgt_width, (layerh_n2,))

#no_w = np.random.normal(0, 1, (layero_n,layerh_n+1))    ## for output layer neuron1,including bias, each row of wgt is used for 1 output neuron
no_w = np.random.normal(0, wgt_width, (layero_n,layerh_n2+1))    ## for output layer neuron1, last one is bias
#b_o = np.random.normal(0, wgt_width, (layero_n,)) #bias--output layer

# learning rate
eta = 0.1

# target output
y=np.diag(np.ones((layero_n,)))
y[0]=0


###############################################
# Epochs
###############################################
traindata=train_dat
epoch =10 # how many epochs?
err = np.zeros((epoch, 1))  # lets record error to plot (get a convergence plot)
inds = np.arange(np.size(traindata,0))  # array of our training indices (data point index references)
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
        #print("data:", i)
        # what index?
        inx = inds[i]
        img_data=traindata[inx,80,0:-1].reshape(img_dim,img_dim)  ### convert to img matrix
    # forward pass
        v = np.ones((feature_n,sliding_o,sliding_o))

        kk=0
        for j in range(feature_n):
            convimg(v,m=img_data,img_dim=img_dim,share_wgt_dim=share_wgt_dim,wgt=nh_w[j,:,:],b=b_h[j],f=j)
        v = acti(v)

        vh2=np.ones((layerh_n2+1,))   ###including bias

        ###from first hidden to sec hidden
        for sss in range(layerh_n2):
            vh2[sss] = np.dot(nh2_w[sss,:,:].flatten(), v.flatten()) + b_h2[sss]  ### or np.multiply(nh2_w[sss,:,:], v).sum()+ b_h2[sss]
            vh2[sss] = acti(vh2[sss])

        ###output layer
        oo = no_w @ vh2
        o = acti(oo)  # result of output 0&1 !!!

        ###calculating error
        err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[inx,:]), 2.0)).sum()

        # backprop time!!! ################################
        # output layer
        # delta_ow = np.zeros((layero_n, layerh_n+1))   ##last col is for bias
        delta_ow = np.zeros((layero_n,layerh_n2+1)) ## last term is for bias
        #delta_ob = np.zeros((layero_n, ))

        #if traindata[inx, -1] <3.0:
            #delta_1 = o - y[0, :]
       # else:
            #delta_1 = o - y[1, :]
        delta_1 = o - y[inx, :]
        delta_2 = acti(o, derive=True)
        #print('delta2',delta_2)

        '''for uuu in range(layero_n):
            for hhh in range(layerh_n+1):
                delta_ow[uuu,hhh]= delta_1[uuu]*delta_2[uuu]*v[hhh]'''
        delta_ow=np.array([np.multiply(delta_1,delta_2)]).T@np.array([vh2])   ###including bias(the last col on each row)

         # for wgt from hidden layer 1-hidden layer2

        delta_3=np.multiply(no_w.T,np.multiply(delta_1, delta_2)).sum(axis=1)

        delta_4 = acti(vh2, derive=True)

        delta_h2=np.multiply(delta_3,delta_4)    ###for wgt from hideen 1-hidden layer2

        delta_hw2=(np.array([delta_h2[:-1]]).T*np.array([v.flatten()])).reshape(layerh_n2,feature_n,sliding_o,sliding_o)      ###for wgt from h1 to h2 (excluding bias)

         ### for last layer

        delta_h0 = acti(v[:, :, :], derive=True)

        delta_hw=(np.array([delta_h2[:-1]]).T*np.array([nh_w.flatten()])).reshape(layerh_n2,feature_n,sliding_o,sliding_o) # need to sum up to 16*22*22 dim

        #delta_4=np.zeros((layero_n, feature_n, sliding_o, sliding_o))

       # for mmo in range(layero_n):  ### loop over output layer
                #backwgt[mmo, :, :, :] = delta_1[mmo] * delta_2[mmo] * no_w[mmo, :, :, :]
            #delta_4[mmo, :, :, :] =  np.multiply(delta_1[mmo] * delta_2[mmo] * no_w[mmo, :, :, :],delta_3)

        delta_nh=np.zeros((feature_n,share_wgt_dim,share_wgt_dim))
        delta_bh=np.zeros((feature_n,))


        #delta_4=np.multiply(delta_3, np.sum(backwgt, 0))
        #for mmo in range(layero_n):
           # delta_4[mmo, :, :, :]=np.multiply(backwgt[mmo, :, :, :],delta_3)

        '''img_input_slice=np.zeros((share_wgt_dim,share_wgt_dim))
        for jup in range(share_wgt_dim):
            for jlr in range(share_wgt_dim):
                img_input_slice[jup,jlr]=img_data[jup:jup + sliding_o, jlr:jlr + sliding_o]'''

        '''for mmf in range(feature_n):
            # print (mmf)
            for mmo in range(layero_n):
                for hup in range(sliding_o):
                     for hlr in range(sliding_o):
                         delta_bh[mmf] += delta_1[mmo] * delta_2[mmo] * no_w[mmo, mmf, hup, hlr] * delta_3[
                             mmf, hup, hlr]
                         for jup in range(share_wgt_dim):
                             for jlr in range(share_wgt_dim):
                                 delta_nh[mmf, jup, jlr] += delta_1[mmo] * delta_2[mmo] * no_w[mmo, mmf, hup, hlr] * \
                                                           delta_3[mmf, hup, hlr] * img_data[hup + jup, hlr + jlr]'''

        for mmo in range(layero_n):
            # print (mmf)
            for mmf in range(feature_n):
                for hup in range(sliding_o):
                    for hlr in range(sliding_o):
                        delta_bh[mmf] += delta_1[mmo] * delta_2[mmo] * no_w[mmo, mmf, hup, hlr] * delta_3[mmf, hup, hlr]
                        for jup in range(share_wgt_dim):
                            for jlr in range(share_wgt_dim):
                                delta_nh[mmf, jup, jlr] += delta_1[mmo] * delta_2[mmo] * no_w[mmo, mmf, hup, hlr] * delta_3[mmf, hup, hlr] * img_data[hup + jup, hlr + jlr]



                #delta_bh[mmf]+=np.sum(delta_4[mmo,mmf,:,:])
                #for jup in range(share_wgt_dim):
                    #for jlr in range(share_wgt_dim):
                        #delta_nh[mmf,jup,jlr] + = np.multiply(delta_4[mmf, :, :],img_data[jup:jup + sliding_o, jlr:jlr + sliding_o]).sum()
                        #delta_nh[mmf, jup, jlr]+=np.sum(np.multiply(delta_4[mmo,mmf,:,:],img_data[jup:jup + sliding_o, jlr:jlr + sliding_o]))
                '''for mm1 in range(feature_n):  ### loop over feature
            for mm2 in range(layero_n):  ### loop over output layer
                delta_bh[mm1] += delta_1[mm2] * delta_2[mm2] * np.multiply(no_w[mm2, mm1, :, :],delta_3[mm1, :, :]).sum()

        for mm1 in range(feature_n):  ### loop over feature
            for mm2 in range(layero_n): ### loop over output layer
                for jup in range(share_wgt_dim):
                    for jlr in range(share_wgt_dim):
                        delta_nh[mm1,jup,jlr]+=delta_1[mm2]*delta_2[mm2]*np.multiply(np.multiply(no_w[mm2,mm1,:,:],delta_3[mm1,:,:]),img_data[jup:jup+sliding_o,jlr:jlr+sliding_o]).sum()'''

                '''for hh in range(feature_n):
            delta_h = acti(v[hh, :, :], derive=True)
            for mm in range(layero_n):
                for hhh in range(sliding_o**2):
                    delta_1[mm] * delta_2[mm] * no_w[mm, hh*sliding_o+hhh]*delta_h[hh,hhh]*img_data[jup:jup+sliding_o,jlr:jlr+sliding_o]

                np.multiply(delta_1,delta_2,no_w[:,0])

            for hh in range(feature_n):
                for mm in range(layero_n):
               np.multiply(np.multiply(delta_1, delta_2),no_w[:,mm+sliding_o**2*hh])

            for cc in range(sliding_o**2):  #per feature
                for mm in range(layero_n):
                    delta_nh[hh,:,:]+=delta_1[mm] * delta_2[mm] * no_w[mm,hh] * delta_h*traindata[inx,0:-1].reshape(img_dim,img_dim)
                    delta_bh[hh]+=delta_1[mm] * delta_2[mm] * no_w[mm,hh]* delta_h'''

        # update rule, so old value + eta weighted version of delta's above!
        nh_w = nh_w + (-1.0) * eta * delta_nh
        b_h = b_h + (-1.0) * eta * delta_bh
        no_w = no_w + (-1.0) * eta * delta_ow
        b_o = b_o + (-1.0) * eta * delta_ob
        print('err',err)

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
        # what index?
     inx = inds[i]
     img_data=traindata[inx,80,0:-1].reshape(img_dim,img_dim)  ### convert to img matrix
    # forward pass
     v = np.ones((feature_n,sliding_o,sliding_o))
     for j in range(feature_n):
         convimg(v,m=img_data,img_dim=img_dim,share_wgt_dim=share_wgt_dim,wgt=nh_w[j,:,:],b=b_h[j],f=j)
     v = acti(v)
        #oo = np.array([np.dot(v.T, no_w),np.dot(v, no_w2)])  # output neuron 0&1 fires, taking hidden neuron 1 and 2 as input
        #vv = np.append(v.flatten(), 1)    ## append for bias
        #oo = no_w @ vv
     oo=np.zeros((layero_n,))
        #o=np.zeros((layero_n,))
     for kk in range(layero_n):
            oo[kk]=np.multiply(no_w[kk,:, :, :], v[:, :, :]).sum()+b_o[kk]
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



