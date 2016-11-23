#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.ion()
import sys
#sys.path.insert(0,'/scratch/ys1297/caffe-installation/video-caffe/python')
import caffe
import numpy as np
import pdb
import cPickle as pkl
from sklearn import preprocessing
import h5py
import copy
import math
#import matplotlib
import h5py
import scipy.io as scipy_io
from pylab import *
def net(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=1024, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=1024, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)
    n.ip3 = L.InnerProduct(n.relu2, num_output=2, weight_filler=dict(type='xavier'))
    n.loss = L.SigmoidCrossEntropyLoss(n.ip3, n.label)
    return n.to_proto()

def vis_square(data,image_name):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data) 
    plt.axis('off')
    plt.savefig(image_name)


if __name__=='__main__':

    sample_num = 5000
    #with open('./auto_train.prototxt', 'w') as f:
    #    f.write(str(net('train.h5list', 50)))
    #with open('auto_test.prototxt', 'w') as f:
    #    f.write(str(net('test.h5list', 20)))

    #caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('c3d_ucf101_my_sovler.prototxt')

    #solver.net.copy_from('/scratch/ys1297/caffe-installation/model/verizon/')
    solver.net.copy_from('/scratch/xf362/c3d_ucf101/model1/test_new_17000.000000')
    #filters = solver.net.params['conv1'][0].data
    #vis_square(filters.transpose(0, 2, 3, 1),'test_image_init.png')
 
    niter = 100000
    test_interval = 1000
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
    print len(test_acc)
    output = zeros((niter, 2, 2))
    # The main solver loop
    with open('c3d_ucf101_my_train_log.txt','a') as txt_file:
    	for it in range(niter):
	    solver.step(1)  # SGD by Caffea	
            #pdb.set_trace()
	    train_loss[it] = solver.net.blobs['loss'].data
	    #print 'train_loss'+ str(train_loss[it])
	    print 'train_loss'+ str(train_loss[it]) + " std :" + str(solver.net.params['fc8'][0].data.std()) +"iteration: " +str(it)
	    #solver.net.params['conv1'][0].data
            if it % 100== 0:
		loss = 0
		for test_it in range(100):
                    solver.test_nets[0].forward()
		    loss += solver.test_nets[0].blobs['loss'].data
		txt_file.write('iteration: '+ str(it)+  ' test_loss: '+str(loss/100) +'\n')		
               # txt_file.close()         
           	solver.net.save('model/test_new_%f'%(it))
     

