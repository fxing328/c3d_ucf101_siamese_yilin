import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from pylab import *
import pdb
import os
import h5py
import re
from sklearn.decomposition import PCA
import cPickle as pickle
import scipy.io as scipy_io
def dictionary(descriptors, N):
	em = cv2.EM(N)
	em.train(descriptors)

        pdb.set_trace()
	return np.float32(em.getMat("means")), \
		np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]

#def image_descriptors(file):
#	img = cv2.imread(file, 0)
#	img = cv2.resize(img, (256, 256))
#	_ , descriptors = cv2.SIFT().detectAndCompute(img, None)
#	return descriptors

def folder_descriptors(folder):
	files = glob.glob(folder + "/*.jpg")
	print("Calculating descriptos. Number of images is", len(files))
	return np.concatenate([image_descriptors(file) for file in files])

def likelihood_moment(x, ytk, moment):	
	x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
	return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
	#pdb.set_trace()
        gaussians, s0, s1,s2 = {}, {}, {}, {}
	samples = zip(range(0, samples.shape[0]), samples[:,])
	
	g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
	#pdb.set_trace()
        for index, x in samples:
		gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

        #for index in range(samples.shape[0]):
	#   for k in range(0,len(weights)):	
        #        gaussians[index] = np.array(multivariate_normal.pdf(samples[index,:],mean=means[k],cov=covs[k]))
        #pdb.set_trace()
	for k in range(0, len(weights)):
		s0[k], s1[k], s2[k] = 0, 0, 0
		for index, x in samples:
			probabilities = np.multiply(gaussians[index].T, weights)
			probabilities = probabilities / np.sum(probabilities)
			probabilities = probabilities.T
                        s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
			s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
			s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)
        #pdb.set_trace()
	return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
	return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
	return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
	s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
	T = len(samples)
	covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
        #pdb.set_trace()
 	a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
	b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
	c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
	fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
	fv = normalize(fv)
	return fv

def generate_gmm(feat, N, respath):
        #for n in range(feat.shape[1]):
	words = np.concatenate([feat[n,:] for n in range(feat.shape[0])]) 
        print("Training GMM of size", N)
	means, covs, weights = dictionary(words, N)
	#Throw away gaussians with weights that are too small:
	th = 1.0 / N
	means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
	covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
	weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

	np.save(respath+"means.gmm", means)
	np.save(respath+"covs.gmm", covs)
	np.save(respath+"weights.gmm", weights)
 	#pdb.set_trace()
        return means, covs, weights

#def get_fisher_vectors_from_folder(feat, gmm):
	#files = glob.glob(folder + "/*.jpg")
#	return np.float32([fisher_vector(feat, *gmm)])

def fisher_features(feat, gmm):
#	folders = glob.glob(folder + "/*")
	#features = {f : get_fisher_vectors_from_folder(f, gmm) for f in folders}      
        #for f in range(feat.shape[1]):
        features = fisher_vector(feat,*gmm)
	return features

#def train(gmm, features):
#	X = features
#	Y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(features)), features)])
#
#	clf = svm.SVC()
#	clf.fit(X, Y)
#	return clf

def success_rate(classifier, features):
	print("Applying the classifier...")
	X = np.concatenate(np.array(features))
	Y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(features)), features)])
	res = float(sum([a==b for a,b in zip(classifier.predict(X), Y)])) / len(Y)
	return res
	
def load_gmm(folder = ""):
	if os.path.exists(folder+'means.gmm.npy'):
           files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
	   #for file_gmm in files:
           means = np.load(folder+files[0])
           covs  =  np.load(folder+files[1])
           weights = np.load(folder+files[2])
           return means,covs,weights #map(lambda file: np.load(file), map(lambda s : folder + "/" , files))
        else:
           return -1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d' , "--dir", help="Directory with images" , default='.')
    parser.add_argument("-g" , "--loadgmm" , help="Load Gmm dictionary", action = 'store_true', default = False)
    parser.add_argument('-n' , "--number", help="Number of words in dictionary" , default=5, type=int)
    args = parser.parse_args()
    return args
def feat_pca(respath, features, pca_component):
    pca = PCA(pca_component,whiten = True)
    X = features
    pca.fit(X)
    X_proj = pca.transform(X)
    pickle.dump(pca,open(respath+"pca_comp.p",'w'))

    return X_proj


from scipy.stats import multivariate_normal

def train_gmm(working_folder,respath):
   videosmp_Num = 4000 # video sample numbers from each scene for training
   #videoNum = 360
   #pdb.set_trace()
   trainlist = 'trainlist01.txt'
   K = 512
   pca_component = 128
   with open(trainlist,'rb') as trainlist:
     trainVideo = trainlist.readlines()
   Shotssmp_Num = videosmp_Num*2 
   features_train = np.zeros((Shotssmp_Num,2048))
   n = 0
   videolistspace = np.linspace(0,len(trainVideo)-1,num=len(trainVideo),dtype=int)
   videolist_idx = np.random.choice(videolistspace,videosmp_Num,replace=False)
   
   for video_idx in videolist_idx.tolist():
     #video_idx_all =  np.random.choice(videolist,1)
     #video_idx = trainVideo[video_idx_all]
     video_line = trainVideo[video_idx]
     video_path = re.split(' ',video_line)[0]
     data1 = h5py.File(working_folder+'/'+video_path[:-4]+'.feat.h5','r')['data']
     shotsNum = data1.shape[0]
     shotsList = np.linspace(0,shotsNum-1,num=shotsNum,dtype=int)
     shot_idx = np.random.choice(shotsList,2,replace=False)
     for i in shot_idx.tolist():
       features_train[n,:] = data1[i,:]
       n = n+1
   features_train_pca = feat_pca(respath,features_train,pca_component)
   pdb.set_trace()
   generate_gmm(features_train_pca, K, respath)



if __name__ == '__main__':

   #args = get_args()
   working_folder = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/ucfdata_h5'
   respath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/ucfres_matrix/'
   testlist = 'newtestlist.txt'
   i = 0
   j = 0
   with open(testlist,'rb') as testlist:
     videolist = testlist.readlines()
   #vidFish_matrix = np.zeros((len(videolist),len(videolist)))
   for l1 in videolist:
      i = i+1
      video_file1 = working_folder + '/' + l1[:-6] + '.feat.h5'
      features_test1 = h5py.File(video_file1,'r')['data']
      for l2 in videolist:
         j = j+1
         video_file2 = working_folder + '/' + l2[:-6] + '.feat.h5'
         features_test2 = h5py.File(video_file2,'r')['data']
         if load_gmm(respath) < 0:
   # fisher vector for training
            train_gmm(working_folder,respath) 
   # make feature dimension reduction by using pca, from 2048 to 128 dimension
         else:
            #pdb.set_trace()
            
            vidFish_matrix = scipy_io.loadmat(respath+'CMucf_fv.mat')['video_pair']
            if vidFish_matrix[i-1,j-1]==0:
               gmm = load_gmm(respath)
               pca = pickle.load(open(respath+'pca_comp.p','r')) 
               features_test_pca1 = pca.transform(features_test1)
               features_test_pca2 = pca.transform(features_test2)
               fisher_features1 = fisher_features(features_test_pca1, gmm)
               fisher_features2 = fisher_features(features_test_pca2, gmm)
               vidFish_matrix[i-1,j-1] = np.dot(fisher_features1, fisher_features2)
               vidFish_matrix[j-1,i-1] = np.dot(fisher_features1, fisher_features2)   
            else:
               pass
               print 'pass'
            print i-1,j-1
            if j == len(videolist):
               j = 0
      if i % 20 == 0: 
        ConMatrix = {}
        ConMatrix['video_pair'] = vidFish_matrix
        scipy_io.savemat(open(respath+'CMucf_fv.mat','wb'),ConMatrix)
      if i == len(videolist):
         i = 0

   ConMatrix = {}
   ConMatrix['video_pair'] = vidFish_matrix
   scipy_io.savemat(open(respath+'CMucf_fv.mat','wb'),ConMatrix) 

