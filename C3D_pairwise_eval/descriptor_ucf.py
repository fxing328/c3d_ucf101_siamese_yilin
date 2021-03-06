import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import sys
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
import os
import re
import cv2
caffe.set_mode_gpu()
net = caffe.Net('c3d_ucf101_siamese_deploy_hdf5.prototxt','/scratch/xf362/yilin_revised_pipeline_siamense/C3D_siamense/model/siamese_44000.000000',caffe.TEST)
#net = caffe.Net('c3d_ucf101_siamese_train_test_hdf5.prototxt','/scratch/xf362/c3d_verizon/model/siamese_10000.000000',caffe.TEST)

def cropImg(img,oriRio=0.24/0.36):
   #oriImg = cv2.imread(img)
    #img = np.swapaxes(img,0,2)
    h = img.shape[0]
    w = img.shape[1]
    # from the middle of the long side (the middle two points)to
    # crop based on the shorter side, then according to the ucf101
    # ratio to crop the other side  
    if h <= w * oriRio:
       crop_ws = w/2-1-int(h/(oriRio*2))
       crop_we = w/2+int(h/(oriRio*2))
       subImg = img[:,crop_ws:crop_we,:]
    else:
       crop_hs = h/2-1-int(w*(oriRio/2))
       crop_he = h/2+int(w*(oriRio/2))
       subImg = img[crop_hs:crop_he,:,:]

    return subImg

def create_tensor(file1,mean_array):
    video_1 = cv2.VideoCapture(file1)
    len_1 = int(video_1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
    tensor_1 = np.zeros([3,len_1,112,112])
    count = 0
    ret = True
    while True:
	ret, frame_1 = video_1.read()
        if frame_1 is not None:
 	    tensor_1[:,count,:,:] = np.swapaxes(cv2.resize(cropImg(frame_1),(112,112)),0,2) - mean_array
            count = count+1
	    print count	
	else:
	    break
    tensor = tensor_1[:,:count,:,:]
    return tensor
def eval_pairwise(tensor_1,tensor_2,net):

    pair_matrix = np.zeros([np.floor(tensor_1.shape[1]/8)-1,np.floor(tensor_2.shape[1]/8)-1])
    print pair_matrix.shape 
    for i in range(pair_matrix.shape[0]):
	for j in range(pair_matrix.shape[1]):
	    net.blobs['data_1'].data[0,:,:,:] = tensor_1[:,i*8:i*8+16,:,:]
  	    net.blobs['data_2'].data[0,:,:,:] = tensor_2[:,j*8:j*8+16,:,:]
            output= net.forward()
	    pair_matrix[i,j] = output['prob'][0][1]
        print i
    pdb.set_trace()
    return pair_matrix
   

if __name__ =='__main__':
    datapath = '/scratch/xf362/UCF-101/'
    tensorpath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/ucfdata_h5/'
    #pdb.set_trace()
    for item in sort(os.listdir(datapath)):
      i = 0
      for videofile in sort(os.listdir(datapath+item)): 
        if re.search('.avi',videofile) is not None:
           print videofile
           i = i+1  
           tensor = create_tensor(datapath+item+'/'+videofile,mean_array = pkl.load(open("meanFile.p","rb")))
           if os.path.exists(tensorpath+item):
              pass
           else:
              os.mkdir(tensorpath+item)
           with  h5py.File(tensorpath+item+'/'+videofile+'.h5','w') as f1_pair:
                f1_pair['data'] = tensor
                f1_pair.close()
           #if i >= 10:
              #pdb.set_trace()
           #   break

#    data1 = h5py.File('v_ApplyEyeMakeup_g01_c01.avi.h5','r')['data']
#    data2 = h5py.File('v_ApplyEyeMakeup_g02_c01.avi.h5','r')['data']
#    pair_matrix = eval_pairwise(np.array(data1),np.array(data2),net)
#    rel = {}
#    rel['pair_matrix'] = pair_matrix
#    scipy_io.savemat(open('ucf_ApplyEyeMakeup_pos.mat','wb'),rel)
#    print 1








