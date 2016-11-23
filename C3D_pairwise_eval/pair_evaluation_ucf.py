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
#net = caffe.Net('c3d_ucf101_siamese_deploy_hdf5.prototxt','/scratch/xf362/yilin_revised_pipeline_siamense/C3D_siamense/model/siamese_44000.000000',caffe.TEST)
#net = caffe.Net('c3d_ucf101_siamese_deploy_hdf5.prototxt','/scratch/xf362/c3d_verizon/model/siamese_7000.000000',caffe.TEST)

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
def feature_extraction(datapath,scene):
    batchsize = 100
    protxt = 'deploy_extract_feature.prototxt'
    pair_matrix = np.zeros([40,40])
    net = caffe.Net(protxt,'/scratch/xf362/yilin_revised_pipeline_siamense/C3D_siamense/model/siamese_44000.000000',caffe.TEST)
    #net = caffe.Net(protxt,'/scratch/xf362/c3d_verizon/model/siamese_7000.000000',caffe.TEST)
    #feat_all = np.zeros((2048,40000))
      
      
    for video_file in sort(os.listdir(datapath+scene)):
      if re.search('.avi.h5',video_file) is not None:
         if os.path.exists(datapath+scene+'/'+video_file[:-6]+'feat.h5'):
            pass
            print 'pass'
         else: 
            data = h5py.File(datapath+scene+'/'+video_file,'r')['data']
            shotsNum = int(np.floor(data.shape[1]/8-1))
            feat = np.zeros((shotsNum,2048))
            data1 = np.zeros([shotsNum,3,16,112,112])
            net.blobs['data_1'].data[:,:,:,:,:] = np.zeros([batchsize,3,16,112,112])
            for i in range(shotsNum):
                  data1[i,:,:,:,:] =  data[:,i*8:i*8+16,:,:]

            if shotsNum > batchsize:
               s = shotsNum/batchsize
               k = 0
               count = 0
               j = 0
               while s:
                 net.blobs['data_1'].data[count,:,:,:,:] = data1[k,:,:,:,:]
                 count = count+1
                 k = k+1
                 if count == batchsize:
                    net.forward()
                    count = 0
                    s = s-1
                    for o in range(batchsize):
                        feat[j,:] = net.blobs['norm'].data[o,:]
                        j = j+1
               pdb.set_trace()
               while k < shotsNum:
                 pdb.set_trace()
                 net.blobs['data_1'].data[count,:,:,:,:] = data1[k,:,:,:,:]
                 k = k+1
                 count = count+1
               pdb.set_trace()
               net.forward()
               for o in range(shotsNum - j):
                 feat[j,:] = net.blobs['norm'].data[o,:]
                 j = j+1    
            else:
               net.blobs['data_1'].data[:shotsNum,:,:,:,:] = data1
               net.forward()
               for j in range(shotsNum):
                   feat[j,:] = net.blobs['norm'].data[j,:]
            
            with h5py.File(datapath+scene+'/'+video_file[:-6]+'feat.h5','w') as f1_pair:
                f1_pair['data'] = feat
                f1_pair.close()
            print video_file[:-6]   
         #print video_file

def pair_eval_prob(video_file1,video_file2):
    batchsize = 100
    protxt = 'MLP_net1.prototxt'
    i = 0
    j = 0
    #k = 0
    #forwordTime = 0
   # pdb.set_trace()
    net = caffe.Net(protxt,'/scratch/xf362/yilin_revised_pipeline_siamense/C3D_siamense/model/siamese_44000.000000',caffe.TEST)
    #net = caffe.Net(protxt,'/scratch/xf362/c3d_verizon/model/siamese_7000.000000',caffe.TEST)
    data1 = h5py.File(video_file1,'r')['data']
    data2 = h5py.File(video_file2,'r')['data']
    data_num1 = np.array(data1).shape[0]
    data_num2 = np.array(data2).shape[0]
    goplist = []
    count = 0
    q = 0
    idx = 0
    pair_matrix = zeros((data_num1,data_num2))
    for m in range(data_num1):
      for n in range(data_num2):
         goplist.append([m,n])
    s = len(goplist)/batchsize
    while s:
      net.blobs['norm'].data[count,:] = data1[goplist[idx][0],:]
      net.blobs['norm_p'].data[count,:] = data2[goplist[idx][1],:]
      count = count+1
      idx = idx+1
      if count == batchsize:
         output = net.forward()
         count = 0
         s = s-1
         for o in range(batchsize):
           pair_matrix[goplist[q][0],goplist[q][1]] = output['prob'][o][1]
           q = q+1
    while idx < len(goplist):
         net.blobs['norm'].data[count,:] = data1[goplist[idx][0],:]
         net.blobs['norm_p'].data[count,:] = data2[goplist[idx][1],:]
         idx = idx+1
         count = count+1
    output = net.forward()
    for k in range(len(goplist)-q):
        pair_matrix[goplist[q][0],goplist[q][1]] = output['prob'][k][1]
        q = q+1
    return pair_matrix


            

if __name__ =='__main__':
#  datapath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/data_h5/'
#  videopath = '/scratch/xf362/verizon_challenge/'
#  for scene_name in sort(os.listdir(datapath)):
#    for item in sort(os.listdir(videopath+scene_name)):
#      if re.search('.mp4',item) is not None:
#         print item  
#         tensor = create_tensor(videopath+scene_name+'/'+item,mean_array = pkl.load(open("meanFile.p","rb")))
#         with  h5py.File(datapath+scene_name+'/'+item+'.h5','w') as f1_pair:
#           f1_pair['data'] = tensor
#           f1_pair.close()
#

## Generate h5 file for each video that contains numbers of 16-frame shots
#  datapath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/data_h5/' 
#  for scene_name in sort(os.listdir(datapath)):
#    for video_file in sort(os.listdir(datapath+scene_name)):
#       pdb.set_trace()
#       if re.search('.mp4',video_file) is not None:
#:q
#          data = h5py.File(datapath+scene_name+'/'+video_file,'r')['data']
#          tensor = np.array(data)
#          shotsNum = int(np.floor(tensor.shape[1]/8-1))
#          with h5py.File(datapath+scene_name+'/'+video_file[:-6]+'16.h5','w') as f1_pair:
#            f1_pair.create_dataset('data_1',(shotsNum,3,16,112,112),dtype='f8')
#            for i in range(shotsNum):
#               f1_pair['data_1'][i,:,:,:,:] =  tensor[:,i*8:i*8+16,:,:]
#            f1_pair.close()
#    print scene_name          
##         
  datapath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/ucfdata_h5/' 
  savepath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/ucfdata_h5/res_testlist01/'
  ## extract features for each 16-frame GOP in each video shot
  #for scene_name in sort(os.listdir(datapath)):
  #scene_name = 'Chinese_new_year_nyc_2016'
    #pair_matrix = feature_extraction(datapath,scene_name)
  
  ## compare two feature similarity (probability) throuth trained MPL network 
  testlist = 'newtestlist.txt'
  i = 0
  j = 0
  with open(testlist,'rb') as testlist:
    videolist = testlist.readlines()
  video_CofMatrix = np.zeros((len(videolist),len(videolist)))
  for l1 in videolist:
      i = i+1
      #vfile1 = re.split(' ',l1)[0]
      #video_file1 = datapath + vfile1[:-4] + '.feat.h5' 
      video_file1 = datapath + l1[:-6] + '.feat.h5'
     # pdb.set_trace()
      for l2 in videolist:
         j = j+1
         #vfile2 = re.split(' ',l2)[0]
         #video_file2 = datapath + vfile2[:-4] + '.feat.h5'
         video_file2 = datapath + l2[:-6] + '.feat.h5'
         video_name1 = re.split('/',video_file1)[-1][:-8]
         video_name2 = re.split('/',video_file2)[-1][:-8]
         if os.path.exists(savepath+video_name1+'_'+video_name2+'.mat'):
            pass
            pair_matrix = scipy_io.loadmat(savepath+video_name1+'_'+video_name2+'.mat')['pair_matrix']
            print 'read'
         else:
            pair_matrix = pair_eval_prob(video_file1,video_file2)
            rel = {}
            rel['pair_matrix'] = pair_matrix
            scipy_io.savemat(open(savepath+video_name1+'_'+video_name2+'.mat','wb'),rel)
            rels = {}
            rels['pair_matrix'] = pair_matrix.T
            scipy_io.savemat(open(savepath+video_name2+'_'+video_name1+'.mat','wb'),rels)
           
         video_CofMatrix[i-1,j-1] = np.mean(pair_matrix)
         if j == len(videolist):
            j = 0   
  pdb.set_trace()
  scene_eval = {}
  scene_eval['video_pair'] = video_CofMatrix
  scipy_io.savemat(open(savepath+'ConfMatrix_'+testlist[:-4]+'.mat','wb'),scene_eval)
  print 1








