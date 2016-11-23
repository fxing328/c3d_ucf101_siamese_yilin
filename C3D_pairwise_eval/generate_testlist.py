import os
#import math
#from sklearn import preprocessing
#import sys
import re
import matplotlib
#import copy
from pylab import *
import pdb
if __name__ =='__main__':
 datapath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/data_h5/'
 listpath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/testlist/'
 for scene_name in sort(os.listdir(datapath)):
    with open(listpath+'testlist_'+scene_name+'.txt','w') as test_file:
      
      for video_name in sort(os.listdir(datapath+scene_name)):
        #pdb.set_trace()
        if re.search('.16.h5',video_name) is not None:
           line_info = datapath + scene_name +'/'+video_name+'\n' 
           test_file.write(line_info)
      test_file.close()
    print scene_name

