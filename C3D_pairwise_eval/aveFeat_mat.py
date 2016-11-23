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
#net = caffe.Net('c3d_ucf101_siamese_deploy_hdf5.prototxt','/scratch/xf362/c3d_verizon/model/siamese_10000.000000',caffe.TEST)
net1 = caffe.Net('MLP_net1.prototxt','/scratch/xf362/yilin_revised_pipeline_siamense/C3D_siamense/model/siamese_44000.000000',caffe.TEST)
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
    return tensor_1
def eval_pairwise(tensor_1,tensor_2,net,net1):
    pair_matrix = np.zeros([np.floor(tensor_1.shape[1]/8)-1,np.floor(tensor_2.shape[1]/8)-1])
    avgFeat1 = np.zeros((2048,np.floor(tensor_1.shape[1]/8)-1))
    avgFeat2 = np.zeros((2048,np.floor(tensor_2.shape[1]/8)-1))
    print pair_matrix.shape
    m = 0
    n = 0
    try: 
    	for i in range(pair_matrix.shape[0]):
            for j in range(pair_matrix.shape[1]):
	        net.blobs['data_1'].data[0,:,:,:,:] = tensor_1[:,i*8:i*8+16,:,:]
  	        net.blobs['data_2'].data[0,:,:,:,:] = tensor_2[:,j*8:j*8+16,:,:]
	        output= net.forward()
                if i == 0:
            	   avgFeat2[:,n] = net.blobs['norm_p'].data[0,:]
                   n = n+1
            avgFeat1[:,m] = net.blobs['norm'].data[0,:]
            m = m+1
   	#pdb.set_trace()
        print '1111'
    except:
        pdb.set_trace()
    pdb.set_trace()
    AvgFeat1 = np.mean(avgFeat1,axis = 1)
    AvgFeat2 = np.mean(avgFeat2,axis = 1)
            #pair_matrix[i,j] = output['prob'][0][1]
    net1.blobs['norm'].data[0,:] = AvgFeat1
    net1.blobs['norm_p'].data[0,:] = AvgFeat2
    output1 = net1.forward()
    sim_res = output1['prob'][0][1]
            #print i
            #pdb.set_trace()
    return sim_res
   

if __name__ =='__main__':
# for item in sort(os.listdir('/scratch/xf362/verizon_challenge/Chinese_new_year_nyc_2016')):
#    if re.search('.mp4',item) is not None:
#       print item  
#    tensor = create_tensor('/scratch/xf362/verizon_challenge/Chinese_new_year_nyc_2016/'+item,mean_array = pkl.load(open("meanFile.p","rb")))
#    with  h5py.File(item+'.h5','w') as f1_pair:
#       f1_pair['data'] = tensor
#       f1_pair.close()
##
    data1 = h5py.File('police_running_far1.mp4.h5','r')['data']
    data2 = h5py.File('arresting_a_man1.mp4.h5','r')['data']
    sim_res = eval_pairwise(np.array(data1),np.array(data2),net,net1)
    #rel = {}
    #rel['pair_matrix'] = pair_matrix
    #scipy_io.savemat(open('Bal_pair_matrix_neg.mat','wb'),rel)
    print sim_res








