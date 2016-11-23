import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

import numpy as np
import scipy.io as scipy_io
import pdb
from  collections import *
import sys
import os
import re
from pylab import *

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
        
    def add_node(self, value):
        self.nodes.add(value)
    
    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        #self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance


def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}
            
    nodes = set(graph.nodes)
                
    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break
    
        nodes.remove(min_node)
        current_weight = visited[min_node]
    
        for edge in graph.edges[min_node]:
            try:
                weight = current_weight + graph.distances[(min_node, edge)]
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node
            except:
                pdb.set_trace()

    return visited, path



def provide_graph(matrix,graph):
    # assuming matrix is fat matrix otherwise transpose the matrix
    #if matrix.shape[0]>= matrix.shape[1]:
    #   pass
    #else:
    #   print "transpose matrix"
    #   return None

    width = matrix.shape[0]
    height = matrix.shape[1]
    

    # add nodes plus source and destination
    for i in range(matrix.size+2):
        graph.add_node(i)
    #add source node connection to the first ones
    for i in range(matrix.shape[0]):
        graph.add_edge(0,i+1,matrix[i,0])

    for j in range(matrix.shape[1]-1):
        for i in range(matrix.shape[0]):
        
            for k in range(matrix.shape[0]):
                if i <=k:
                    graph.add_edge(j*width+i+1,(j+1)*width+k+1,matrix[k,j+1])
                else :
                    graph.add_edge(j*width+i+1,(j+1)*width+k+1,1000000000)

    for i in range(matrix.shape[0]):
        graph.add_edge(matrix.size+1-i,matrix.size+1,0)
    return graph

def optimal_path(matrix):
    
    #matrix = scipy_io.loadmat(sys.argv[1])['pair_matrix'].T
    
    original_matrix = matrix
    
    
    # to avoid infinity error
    matrix = -np.log(matrix +0.00001)
    
    graph_example = Graph()
    # construct the DAG graph
    graph_example = provide_graph(matrix,graph_example)
    # get the optimal path
    visited,path = dijsktra(graph_example,0)

    node = matrix.size+1
    # backwards throw out the optimal path
    node_list = []
    while node !=0:
        node =  path[node]
        print node
        node_list.append(node)

    matrix_index= np.zeros(matrix.shape)

    value = 0
    for i in range(1,len(node_list)):
        x = np.floor(node_list[i]/matrix.shape[0])
        y = node_list[i] - matrix.shape[0]*x
        print 'x:' + str(x) + 'y:' + str(y)
        matrix_index[int(y),int(x)] = 1
        value+= matrix[int(y),int(x)]
    
    value = np.exp(-(1.0/matrix.shape[1])*value)
   # plt.subplot(2,1,1)
   # plt.imshow(original_matrix)
   # plt.subplot(2,1,2)
   # plt.imshow(matrix_index)
   # plt.savefig(image_name)
    return value

if __name__ == '__main__':

    videopath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/data_h5/'
    datapath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/verify_matrix/'
    
    respath = '/scratch/xf362/yilin_revised_pipeline_siamense/C3D_pairwise_eval/res_matrix/'
  #for scene_name in sort(os.listdir(datapath)):
    scene_name = 'Chinese_new_year_nyc_2016'
    video_pair = np.zeros((40,40))
    i = 0
    j = 0
    #pdb.set_trace()
    for video_file1 in sort(os.listdir(videopath+scene_name)):
      if re.search('.mp4',video_file1) is not None:
         i = i+1
         video_name1 = video_file1[:-7]
         for video_file2 in sort(os.listdir(videopath+scene_name)):
           if re.search('.mp4',video_file2) is not None:
              j = j+1
              video_name2 = video_file2[:-7]
              pair_matrix = scipy_io.loadmat(datapath+scene_name+'/'+video_name1+'_'+video_name2+'.mat')['pair_matrix']
              #pair_matrix2 = scipy_io.loadmat(datapath+scene_name+'/'+video_name2+'_'+video_name1+'.mat')['pair_matrix']
              #pair_matrix = (pair_matrix1+pair_matrix2.T)/2
                  #return None
	      #image_name ='./tmp/'+ str(i)+'_'+str(j)+'.pdf'
              #pdb.set_trace()
             # if pair_matrix.shape[0]>= pair_matrix.shape[1]:
             #     pass
             # else:
             #     pair_matrix = pair_matrix.T
              #rowmax = 0
              #
              #for m in range(pair_matrix.shape[0]):
              #    rowmax = pair_matrix[m,:].max()+rowmax
              #max_output = rowmax/pair_matrix.shape[0]
              #video_pair[i-1,j-1] = rowmax
              perc_max = 0
              matrix_idx = np.array(np.where(pair_matrix >= np.percentile(pair_matrix,50)))
              for m in range(matrix_idx.shape[1]):
                 perc_max = perc_max + pair_matrix[matrix_idx[:,m][0],matrix_idx[:,m][1]]
              perc_maxoutput = perc_max/matrix_idx.shape[1]   
              video_pair[i-1,j-1] = perc_maxoutput
              if j == 40:
                 j = 0
    res = {}
    res['video_pair'] = video_pair
    scipy_io.savemat(open(respath+scene_name+'_50percentmaxmean.mat','wb'),res) 
    print 1





