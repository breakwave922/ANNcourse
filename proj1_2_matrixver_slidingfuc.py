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
share_wgt_dim=19 ## share wgt dim
feature_n=2    ## feature map number
sliding_o=img_dim-share_wgt_dim+1  ## sliding output size
layerh_n=sliding_o**2*feature_n   ## #of neurons in hidden layer
layero_n=2   ## # of neurons in output layer
#imagesc( reshape(Matrix1_Test(:,1),[28 28]) )
#plt.imshow(train13_dat[0,0:-1].reshape(img_dim,img_dim))

#initialize wgt
nh_w = np.random.normal(0, 1, (feature_n,share_wgt_dim,share_wgt_dim))   ##hidden layer feature maps
#n2_w = np.random.normal(0, 1, (img_dim,img_dim))   ##hidden layer feature maps-2
b_h = np.random.normal(0, 1, (feature_n,)) #bias
#b_h2 = np.random.normal(0, 1, (1,)) #bias

#no_w = np.random.normal(0, 1, (layero_n,layerh_n+1))    ## for output layer neuron1,including bias, each row of wgt is used for 1 output neuron
no_w = np.random.normal(0, 1, (layero_n,feature_n,sliding_o,sliding_o))    ## for output layer neuron1,
b_o = np.random.normal(0, 1, (layero_n,)) #bias--output layer
no_w[0,0,0,0]=-0.29843791
no_w[0,1,0,0]=-2.14979213
no_w[1,0,0,0]=0.38631747
no_w[1,1,0,0]=0.2879163
b_o[0]=0.40241239
b_o[1]=1.59231296

# learning rate
eta = 2

# target output
#y=np.diag(np.ones((layero_n,)))
y=np.array([[1,0],[0,1]])  ##first/sec row corrsponds to class1&2



###############################################
# Epochs
###############################################
traindata=train13_dat
epoch = 300 # how many epochs?
err = np.zeros((epoch, 1))  # lets record error to plot (get a convergence plot)
inds = np.arange(np.size(traindata,0))  # array of our training indices (data point index references)
inds=[25]
#inds=np.array([0])
#f = IntProgress(min=0, max=epoch)  # instantiate the bar (you can "see" how long alg takes to run)
#display(f)  # display the bar!

for k in range(epoch):
    print(k)
    # init error
    err[k] = 0

    # random shuffle of data each epoch
    # inds = np.random.permutation(inds)
    for i in range(np.size(inds)):
        # what index?
        inx = inds[i]
        img_data=traindata[inx,0:-1].reshape(img_dim,img_dim)  ### convert to img matrix
    # forward pass
        v = np.ones((feature_n,sliding_o,sliding_o))
        #v = np.ones((feature_n*sliding_o*sliding_o,1))
        kk=0
        for j in range(feature_n):
            convimg(v,m=img_data,img_dim=img_dim,share_wgt_dim=share_wgt_dim,wgt=nh_w[j,:,:],b=b_h[j],f=j)
            '''for jup in range(img_dim - share_wgt_dim + 1):  # scan from up to down
                for jlr in range(img_dim - share_wgt_dim + 1):  # scan from left to right
                    v[kk] = np.multiply(traindata[inx,0:-1].reshape(img_dim,img_dim)[jup:jup+share_wgt_dim,jlr:jlr+share_wgt_dim],nh_w[j,:,:]).sum()+b_h[j]
                    v[kk] = acti(v[kk])
                    kk+=1'''
        #v=v.reshape(feature_n,sliding_o,sliding_o)
        v = acti(v)
        #oo = np.array([np.dot(v.T, no_w),np.dot(v, no_w2)])  # output neuron 0&1 fires, taking hidden neuron 1 and 2 as input
        #vv = np.append(v.flatten(), 1)    ## append for bias
        #oo = no_w @ vv
        oo=np.zeros((layero_n,))
        #o=np.zeros((layero_n,))
        for kk in range(layero_n):
            oo[kk]=np.multiply(no_w[kk,:, :, :], v[:, :, :]).sum()+b_o[kk]
        o = acti(oo)  # result of output 0&1 !!!
        print("output:", o)

         ###calculating error
        if traindata[inx,-1]<3.0:
            err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[0,:]), 2.0)).sum()
        else:
            err[k] = err[k] + ((1.0 / 2.0) * np.power((o - y[1,:]), 2.0)).sum()

         # backprop time!!! ################################
         # output layer
         # delta_ow = np.zeros((layero_n, layerh_n+1))   ##last col is for bias
        delta_ow = np.zeros((layero_n, feature_n, sliding_o, sliding_o))
        delta_ob = np.zeros((layero_n,))

        if traindata[inx, -1] < 3.0:
            delta_1 = o - y[0, :]
        else:
            delta_1 = o - y[1, :]
        delta_2 = acti(o, derive=True)

        '''for uuu in range(layero_n):
            for hhh in range(layerh_n+1):
                delta_ow[uuu,hhh]= delta_1[uuu]*delta_2[uuu]*v[hhh]'''
        #print(delta_1,delta_2)
        for uuu in range(layero_n):
            delta_ow[uuu, :, :, :] = delta_1[uuu] * delta_2[uuu] * v[:, :, :]
            delta_ob[uuu] = delta_1[uuu] * delta_2[uuu]


            # for hidden layer
        # backwgt = np.zeros((layero_n, feature_n, sliding_o, sliding_o))

        delta_3 = acti(v[:, :, :], derive=True)

        delta_nh = np.zeros((feature_n, share_wgt_dim, share_wgt_dim))
        delta_bh = np.zeros((feature_n,))



        '''img_input_slice=np.zeros((share_wgt_dim,share_wgt_dim))
        for jup in range(share_wgt_dim):
            for jlr in range(share_wgt_dim):
                img_input_slice[jup,jlr]=img_data[jup:jup + sliding_o, jlr:jlr + sliding_o]'''

        for mmo in range(layero_n):
            # print (mmf)
            for mmf in range(feature_n):
                for hup in range(sliding_o):
                    for hlr in range(sliding_o):
                        delta_bh[mmf] += delta_1[mmo] * delta_2[mmo] * no_w[mmo, mmf, hup, hlr] * delta_3[mmf, hup, hlr]
                        for jup in range(share_wgt_dim):
                            for jlr in range(share_wgt_dim):
                                delta_nh[mmf, jup, jlr] += delta_1[mmo] * delta_2[mmo] * no_w[mmo, mmf, hup, hlr] * \
                                                           delta_3[mmf, hup, hlr] * img_data[hup + jup, hlr + jlr]

        # update rule, so old value + eta weighted version of delta's above!
        nh_w = nh_w + (-1.0) * eta * delta_nh
        b_h = b_h + (-1.0) * eta * delta_bh
        no_w = no_w + (-1.0) * eta * delta_ow
        b_o = b_o + (-1.0) * eta * delta_ob
        #print(err)

# plot it
plt.plot(err)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('Convergence Plot')
plt.show()
#print(err)

# run through out data, what should label be, what did we get?
#inds = np.random.permutation(inds)
# random shuffle of data each epoch
    # inds = np.random.permutation(inds)
for i in range(np.size(inds)):
        # what index?
     inx = inds[i]
     img_data=traindata[inx,0:-1].reshape(img_dim,img_dim)  ### convert to img matrix
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
     print("Sample " + str(i) + ": label " + str(traindata[inx,-1]) + ": got " + str(o))


#### using final trained wgt to test
testdata=train13_dat
#inds = np.arange(np.size(testdata,0))  # array of our training indices (data point index references)
#inds = np.random.permutation(inds)
inds=np.arange(1)
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
      o = acti(oo)  # result of output 0&1 !!!


      #print("Sample " + str(i) + ": label " + str(testdata[inx,-1]) + ": got " + str(o))

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


