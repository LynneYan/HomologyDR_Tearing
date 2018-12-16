import time
import sys
import numpy as np
import km
import json
import os
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import Isomap
from sklearn import preprocessing
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path

import file_util
import mapper_util

_DATA_DIR = '../Data'
n_components = 2
n_neighbors = 8

def generate_cut_graph(X, kng, sgn_info, lim_info):
    sgn = sgn_info[0][0]
    sgn_dirc = sgn_info[0][1]
    sgn_val = sgn_info[0][2]
    
    lim_dirc = lim_info[0]
    lim_val = lim_info[1]
    
    if len(sgn_info)==1:
        print ("Tearing Graph...")
        for i in range (0, len(X)):
            for j in range (0, len(X)):
                if kng[i, j]>0:
                    if sgn>0:
                        if ((X[i, lim_dirc]>=lim_val and X[j, lim_dirc]<lim_val) or (X[i, lim_dirc]<=lim_val and X[j, lim_dirc]>lim_val)) and (X[i, sgn_dirc] > sgn_val ):
                            kng[i, j]=0
                    else:
                        if ((X[i, lim_dirc]>=lim_val and X[j, lim_dirc]<lim_val) or (X[i, lim_dirc]<=lim_val and X[j, lim_dirc]>lim_val)) and (X[i, sgn_dirc] < sgn_val ):
                            kng[i, j]=0
    if len(sgn_info)==2:
        sgn2 = sgn_info[1][0]
        sgn_dirc2 = sgn_info[1][1]
        sgn_val2 = sgn_info[1][2]
        print ("Tearing Graph...")
        for i in range (0, len(X)):
            for j in range (0, len(X)):
                if kng[i, j]>0:
                    if sgn>0 and sgn2>0:
                        if ((X[i, lim_dirc]>=lim_val and X[j, lim_dirc]<lim_val) or (X[i, lim_dirc]<=lim_val and X[j, lim_dirc]>lim_val)) and (X[i, sgn_dirc] > sgn_val )and (X[i, sgn_dirc2] > sgn_val2 ):
                            kng[i, j]=0
                    elif sgn<0 and sgn2>0:
                        if ((X[i, lim_dirc]>=lim_val and X[j, lim_dirc]<lim_val) or (X[i, lim_dirc]<=lim_val and X[j, lim_dirc]>lim_val)) and (X[i, sgn_dirc] < sgn_val )and (X[i, sgn_dirc2] > sgn_val2 ):
                            kng[i, j]=0
                    elif sgn>0 and sgn2<0:
                        if ((X[i, lim_dirc]>=lim_val and X[j, lim_dirc]<lim_val) or (X[i, lim_dirc]<=lim_val and X[j, lim_dirc]>lim_val)) and (X[i, sgn_dirc] > sgn_val )and (X[i, sgn_dirc2] < sgn_val2 ):
                            kng[i, j]=0

                    else:
                        if ((X[i, lim_dirc]>=lim_val and X[j, lim_dirc]<lim_val) or (X[i, lim_dirc]<=lim_val and X[j, lim_dirc]>lim_val)) and (X[i, sgn_dirc] < sgn_val )and (X[i, sgn_dirc2] < sgn_val2 ):
                            kng[i, j]=0
                            
    print ("Rebuild graph...")
    dist_matrix = graph_shortest_path(kng,
                                      method='auto',
                                      directed=False)

    return dist_matrix


def _projection(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    for dataset in _DATA_TYPE:
        for noise in noise_:
            for outlier in outlier_:
                data_dir_ = os.path.join(_DATA_DIR, dataset+'_'+str(noise)+'_'+str(outlier))
                plot_dir_ =  os.path.join(data_dir_,'Plot')
                file_util.make_dir(plot_dir_)
                plot_dir =  os.path.join(plot_dir_,'Projection')
                file_util.make_dir(plot_dir)
                if noise == 0 and outlier ==0:
                    title = dataset
                    n_neighbors = 8
                if noise >0:
                    title = dataset +" with noise"
                    n_neighbors = 8
                if outlier >0:
                    title = dataset +" with outliers"
                    n_neighbors = 8
                print ("------------------"+title+"--------------------------")
                for exp in range (1, exps+1):
                    print ("----------Exp_"+str(exp)+"----------------")
                    data_dir = os.path.join(data_dir_, 'Exp_'+str(exp))
                    Json_dir = os.path.join(data_dir, 'Json')
                    Projection_dir = os.path.join(data_dir, 'Projection')
                    file_util.make_dir(Projection_dir)
                
                    print ("Loading data set...")
                    sys.stdout.flush()
                    X, color = file_util._get_normalized_input_data(data_dir)
                    in_file_json = os.path.join(Json_dir, "Y_base_point_geodesic_distance.json")
                    ninst, dim = X.shape
                    print ("Done.")
                    color_file = os.path.join(Json_dir, "Y_base_point_geodesic_distance.txt")
                    color = np.loadtxt(color_file)
                               
                    Landmark, links = mapper_util._compute_skeleton(in_file_json, X)
                    
                    Index_of_landmarks = []
                    for i in range(0, len(Landmark)):
                        Index = [ x for x, y in enumerate(X) if y[0] == Landmark[i, 0] and  y[1] == Landmark[i, 1] and  y[2] == Landmark[i, 2]]
                        Index_of_landmarks.append(Index[0])

                    start_time = time.time()
                    sample_size = len(Index_of_landmarks)
    
                    sample_ = np.random.permutation(ninst)
                    sample = sample_[range(sample_size)]


                    Y, proj_landmark = Isomap.Landmark_Isomap(X, 2, sample, None)
                    #Y = Isomap.Landmark_Isomap(X, 2, sample, None)
                    end_time = time.time() - start_time
                    out_file = os.path.join(Projection_dir,"Y_random.txt")
                    
                    fig = plt.figure(figsize=(10,10))
                    ax = fig.add_subplot(311)
                    ax.scatter(Y[:,0],Y[:, 1], s= 3, c = color,cmap=plt.cm.get_cmap('coolwarm'))
                    plt.title("Dataset: "+title+"; LIsomap: random-based; Data Size: "+str(len(Y))+"; Landmark size: "+ str(sample_size))
                    ax.scatter(proj_landmark[:, 0], proj_landmark[:, 1], c='black', s=100,  marker=(5, 1),label='Landmarks(Random)')
                    ax.legend()

		    Y = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(Y)
		    np.savetxt(out_file, Y.real, fmt ="%f")
              
                    start_time = time.time()
                    Y, proj_landmark = Isomap.Landmark_Isomap(X, 2, Index_of_landmarks, None)
                    end_time = time.time() - start_time
                    out_file = os.path.join(Projection_dir,"Y_Mapper.txt")
    
                    ax = fig.add_subplot(312)
                    ax.scatter(Y[:,0].real,Y[:, 1].real,s=3, c = color,cmap=plt.cm.get_cmap('coolwarm'))
                    ax.scatter(proj_landmark[:, 0], proj_landmark[:, 1], c='black', s=100,  marker=(5, 1),label='Landmarks(Homology-Preserving)')

                    link_num = len(links)
                    for i in range (0, link_num):
                        ax.plot( [proj_landmark[links[i][0]][0], proj_landmark[links[i][1]][0]], [proj_landmark[links[i][0]][1],proj_landmark[links[i][1]][1]],color = 'black')

                    ax.legend()   
                    plt.title("Dataset: "+title+"; MLIsomap: Mapper-based; Data Size: "+str(len(Y))+"; Landmark size: "+ str(len(Landmark)))

		    Y = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(Y)
		    np.savetxt(out_file, Y.real, fmt ="%f")

                    start_time = time.time()
                    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
                    end_time = time.time() - start_time
                    out_file = os.path.join(Projection_dir,"Y_Isomap.txt")
                    
                    ax = fig.add_subplot(313)
                    ax.scatter(Y[:,0],Y[:, 1], s=3, c = color,cmap=plt.cm.get_cmap('coolwarm'))
                    plt.title("Dataset: "+title+"; Isomap: classic")
                    plt.show()
                    
		    Y = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(Y)
		    np.savetxt(out_file, Y.real, fmt ="%f")
                    plot_file = os.path.join(plot_dir, "Exp_"+str(exp)+".png")
                    fig.savefig(plot_file)

def _projection_R(_DATA_TYPE):
    for dataset in _DATA_TYPE:
        data_dir = os.path.join(_DATA_DIR, dataset)
        Json_dir = os.path.join(data_dir, 'Json')
        Projection_dir = os.path.join(data_dir, 'Projection')
        file_util.make_dir(Projection_dir)
        plot_dir =  os.path.join(data_dir,'Plot')
        
        if dataset == 'octa':
            filter_function = "eccentricity"
        if dataset == 'airfoil1':
            filter_function = "base_point_geodesic_distance"
                 
        print ("Loading data set...")
        sys.stdout.flush()
        infile = os.path.join(data_dir, 'X.txt')
        X = np.loadtxt(infile)
        in_file_json = os.path.join(Json_dir, 'Y_'+filter_function+'.json')
        ninst, dim = X.shape
        print ("Done.")
        color_file = os.path.join(Json_dir, 'Y_'+filter_function+'.txt') 
        color = np.loadtxt(color_file)
                               
        Landmark, links = mapper_util._compute_skeleton(in_file_json, X)
                    
        Index_of_landmarks = []
        for i in range(0, len(Landmark)):
            Index = [ x for x, y in enumerate(X) if y[0] == Landmark[i, 0] and  y[1] == Landmark[i, 1] and  y[2] == Landmark[i, 2]]
            Index_of_landmarks.append(Index[0])

        start_time = time.time()
        sample_size = len(Index_of_landmarks)
    
        sample_ = np.random.permutation(ninst)
        sample = sample_[range(sample_size)]


        Y, proj_landmark = Isomap.Landmark_Isomap(X, 2, sample, None)
        #Y = Isomap.Landmark_Isomap(X, 2, sample, None)
        end_time = time.time() - start_time
        out_file = os.path.join(Projection_dir,"Y_random.txt")
                    
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(131)
        ax.scatter(Y[:,0],Y[:, 1], s= 1, c = color,cmap=plt.cm.get_cmap('coolwarm'))
        plt.title("Dataset: "+dataset+"; LIsomap: random-based; Data Size: "+str(len(Y))+"; Landmark size: "+ str(sample_size))
        plt.axis("equal")
        ax.scatter(proj_landmark[:, 0], proj_landmark[:, 1], c='black', s=50,  marker=(5, 1),label='Landmarks(Random)')
        ax.legend()
                    
	Y = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(Y)
	np.savetxt(out_file, Y.real, fmt ="%f")
              
        start_time = time.time()
        Y, proj_landmark = Isomap.Landmark_Isomap(X, 2, Index_of_landmarks, None)
        end_time = time.time() - start_time
        out_file = os.path.join(Projection_dir,"Y_Mapper.txt")
    
        ax = fig.add_subplot(132)
        ax.scatter(Y[:,0].real,Y[:, 1].real,s=1, c = color,cmap=plt.cm.get_cmap('coolwarm'))
        ax.scatter(proj_landmark[:, 0], proj_landmark[:, 1], c='black', s=50,  marker=(5, 1),label='Landmarks(Homology-Preserving)')

        link_num = len(links)
        for i in range (0, link_num):
            ax.plot( [proj_landmark[links[i][0]][0], proj_landmark[links[i][1]][0]], [proj_landmark[links[i][0]][1],proj_landmark[links[i][1]][1]],color = 'black')

        ax.legend()
        plt.axis("equal")
        plt.title("Dataset: "+dataset+"; MLIsomap: Mapper-based; Data Size: "+str(len(Y))+"; Landmark size: "+ str(len(Landmark)))

	Y = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(Y)
	np.savetxt(out_file, Y.real, fmt ="%f")

        start_time = time.time()
        Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
        end_time = time.time() - start_time
        out_file = os.path.join(Projection_dir,"Y_Isomap.txt")
                    
        ax = fig.add_subplot(133)
        ax.scatter(Y[:,0],Y[:, 1], s=1, c = color,cmap=plt.cm.get_cmap('coolwarm'))
        plt.title("Dataset: "+dataset+"; Isomap: classic")
        plt.axis("equal")
        plt.show()
                    
	Y = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(Y)
	np.savetxt(out_file, Y.real, fmt ="%f")
        plot_file = os.path.join(plot_dir, "projection.png")
        fig.savefig(plot_file)

def _projection_T(_DATA_TYPE):
    n_neighbors = 12
    n_jobs = 1
    for dataset in _DATA_TYPE:
        data_dir = os.path.join(_DATA_DIR, dataset)
        Json_dir = os.path.join(data_dir, 'Json')
        Projection_dir = os.path.join(data_dir, 'Projection')
        file_util.make_dir(Projection_dir)
        plot_dir =  os.path.join(data_dir,'Plot')
        
        if dataset.startswith("cylinder"):
            filter_function = "width"
                 
        print ("Loading data set...")
        sys.stdout.flush()
        infile = os.path.join(data_dir, 'X.txt')
        X = np.loadtxt(infile)
        in_file_json = os.path.join(Json_dir, 'Y_'+filter_function+'.json')
        ninst, dim = X.shape
        print ("Done.")
        color_file = os.path.join(Json_dir, 'Y_'+filter_function+'.txt') 
        color = np.loadtxt(color_file)
                               
        Landmark, links = mapper_util._compute_skeleton(in_file_json, X)
        link_num = len(links)
        Index_of_landmarks = []
        for i in range(0, len(Landmark)):
            Index = [ x for x, y in enumerate(X) if y[0] == Landmark[i, 0] and  y[1] == Landmark[i, 1] and  y[2] == Landmark[i, 2]]
            Index_of_landmarks.append(Index[0])

        Y = manifold.Isomap(n_neighbors, 2).fit_transform(X)
        out_file = os.path.join(Projection_dir,"Y_Isomap.txt")
        np.savetxt(out_file, Y, fmt="%f")
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(221)
        proj_landmark = Y[Index_of_landmarks, :]
        ax.scatter(Y[:, 0], Y[:, 1], s=10, c=color,cmap=plt.cm.get_cmap('coolwarm'), alpha=0.7)
 
        ax.scatter(proj_landmark[:, 0],proj_landmark[:, 1],c='black', s=70,  marker=(5, 1),label='Landmarks(Homology-Preserving)')
        for i in range (0, link_num):
            if abs(proj_landmark[links[i][0]][0]-proj_landmark[links[i][1]][0]) <100:
                ax.plot( [proj_landmark[links[i][0]][0], proj_landmark[links[i][1]][0]], [proj_landmark[links[i][0]][1],proj_landmark[links[i][1]][1]],color = 'black')
        plt.legend()
        plt.title('Isomap without Tearing')        
        plt.axis('equal')


        tids = [ [[[1, 0, 0.5],[1, 2, 1]], [1, 0]], [[[-1, 0, 0]] , [1, 0]], [[[1, 0, 0]] , [1, 0]]]
        pltid = 2
        for tid in tids:
            nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
            nbrs_.fit(X)
            kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs).toarray()
            dist_matrix = generate_cut_graph(X, kng, tid[0], tid[1])

            Y = Isomap.Tearing_Isomap(dist_matrix,2)
            proj_landmark = Y[Index_of_landmarks, :]
        
            out_file = os.path.join(Projection_dir,"Y_Tearing_"+str(pltid-1)+".txt")
            np.savetxt(out_file, Y, fmt="%f")
            ax = fig.add_subplot(220+pltid)
            ax.scatter(Y[:, 0], Y[:, 1], s =10, c=color,cmap=plt.cm.get_cmap('coolwarm'), alpha=0.7)
            ax.scatter(Y[Index_of_landmarks, 0], Y[Index_of_landmarks, 1],c='black', s=70,  marker=(5, 1),label='Landmarks(Homology-Preserving)')
            for i in range (0, link_num):
                if abs(proj_landmark[links[i][0]][0]-proj_landmark[links[i][1]][0]) <2:
                    ax.plot( [proj_landmark[links[i][0]][0], proj_landmark[links[i][1]][0]], [proj_landmark[links[i][0]][1],proj_landmark[links[i][1]][1]],color = 'black')
            plt.legend()
            plt.axis('equal')
            plt.title('Isomap with Tearing '+str(pltid-1)) 
            pltid +=1
        plt.show()
        plot_file = os.path.join(plot_dir, "tearing.png")
        fig.savefig(plot_file)

