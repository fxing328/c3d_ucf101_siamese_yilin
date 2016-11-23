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
        pdb.set_trace()
	em = cv2.EM(N)
	em.train(descriptors)

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

def generate_gmm(feat, N):
        #for n in range(feat.shape[1]):
	words = np.concatenate([feat[:,n] for n in range(feat.shape[1])]) 
	#pdb.set_trace()
        print("Training GMM of size", N)
	means, covs, weights = dictionary(words, N)
	#Throw away gaussians with weights that are too small:
	th = 1.0 / N
	means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
	covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
	weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

	np.save("means.gmm", means)
	np.save("covs.gmm", covs)
	np.save("weights.gmm", weights)
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
	if os.path.exists('means.gmm.npy'):
           files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
	   #for file_gmm in files:
           means = np.load(folder+'/'+files[0])
           covs  =  np.load(folder+'/'+files[1])
           weights = np.load(folder+'/'+files[2])
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
def feat_pca(features, pca_component):
    pca = PCA(pca_component,whiten = True)
    X = features.T
    pca.fit(X)
    X_proj = pca.transform(X)
    pickle.dump(pca,open("pca_comp.p",'w'))

    return X_proj


from scipy.stats import multivariate_normal

def train_gmm(working_folder,oridata_folder):
   videosmp_Num = 7200 # video sample numbers from each scene for training
   videoNum = 360
   videolist = np.linspace(0,videoNum-1, num=videoNum, dtype=int)
   np.random.shuffle(videolist)
   features_train = np.zeros((2048,videosmp_Num))
   K = 512
   pca_component = 128
   # for scene_name in sort(os.listdir(datapath)): 
   #   for vfeat_file in sort(os.listdir(datatpath+scene)):
   #       if re.search('.feat.h5',vfeat_file) is not None:
   #pdb.set_trace()

   n = 0
   while True:
     video_idx_all =  np.random.choice(videoNum,1)
     scene_idx = video_idx_all/40
     scene_name = sort(os.listdir(working_folder))[scene_idx][0]
     video_idx = video_idx_all%40
     video_name = scene_list[scene_name][video_idx]
     
     data1 = h5py.File(working_folder+'/'+scene_name+'/'+video_name[:-4]+'.feat.h5','r')['data']
     shotsNum = data1.shape[0]
     shot_idx = np.random.random_integers(0,shotsNum-1)
     features_train[:,n] = data1[shot_idx,:]
     n = n+1
     if n >= videosmp_Num:
        break
   features_train_pca = feat_pca(features_train,pca_component)
   #pdb.set_trace()
   generate_gmm(features_train_pca, K)



if __name__ == '__main__':

   #args = get_args()
   working_folder = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/data_h5'
   oridata_folder = '/scratch/xf362/verizon_challenge'
   respath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/res_matrix/fv_matrix'
   scene_list = {}
   video_list = []
   i = 0
   j = 0
   for scene_name in sort(os.listdir(oridata_folder)):
       video_list = []
       for video_file in sort(os.listdir(oridata_folder+'/'+scene_name)):
           if re.search('.mp4', video_file) is not None:
              video_list.append(video_file)
       scene_list[scene_name] = video_list
   
   for scene_name in sort(scene_list.keys()).tolist():
     if os.path.exists(respath+scene_name+'_fv.mat'):
         pass
         print 'pass'
     else:  
         vidFish_matrix = np.zeros((40,40))
         for video_file1 in scene_list[scene_name]:
           features_test1 = h5py.File(working_folder+'/'+scene_name+'/'+video_file1[:-4]+'.feat.h5','r')['data']
           i = i+1
           for video_file2 in scene_list[scene_name]:
               features_test2 = h5py.File(working_folder+'/'+scene_name+'/'+video_file2[:-4]+'.feat.h5','r')['data']
               j = j+1
               if load_gmm(working_folder[:-7]) < 0:
   # fisher vector for training
                  train_gmm(working_folder,scene_list) 
   # make feature dimension reduction by using pca, from 2048 to 128 dimension
               else:
                  #pdb.set_trace()
                  gmm = load_gmm(working_folder[:-7])
                  pca = pickle.load(open(working_folder[:-7]+'pca_comp.p','r')) 
                  features_test_pca1 = pca.transform(features_test1)
                  features_test_pca2 = pca.transform(features_test2)
                  fisher_features1 = fisher_features(features_test_pca1, gmm)
                  fisher_features2 = fisher_features(features_test_pca2, gmm)
                  vidFish_matrix[i-1,j-1] = np.dot(fisher_features1, fisher_features2)
                  print i-1,j-1
                  if j == 40:
                     j = 0
           if i == 40:
              i = 0

         ConMatrix = {}
         ConMatrix['video_pair'] = vidFish_matrix
         scipy_io.savemat(open(respath+scene_name+'_fv.mat','wb'),ConMatrix) 
         print scene_name

