import os
import subprocess
import file_util
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from sklearn import preprocessing

_DATA_DIR = '../Data'
n_components = 2

def _plot_evaluation(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    for dataset in _DATA_TYPE:
        print ("------------------"+dataset+"--------------------------")
        for noise in noise_:
            for outlier in outlier_:
                data_dir_ = os.path.join(_DATA_DIR, dataset+'_'+str(noise)+'_'+str(outlier))
                plot_dir =  os.path.join(data_dir_,'Plot')
                if noise == 0 and outlier ==0:
                    title = dataset
                if noise >0:
                    title = dataset +" with noise"
                if outlier >0:
                    title = dataset +" with outliers"

                dist_Mapper_1 = []
                dist_random_1 = []
                dist_Isomap_1 = []
             
                rv_Mapper = []
                rv_random = []
                rv_Isomap = []
         
            
                for exp in range (1, exps+1):
                    data_dir = os.path.join(data_dir_, 'Exp_'+str(exp))
                    PD_dir = os.path.join(data_dir, 'PD')
                    
     		    if os.path.isfile(os.path.join(PD_dir,"Y_Mapper_norm_dim_1_wd.txt")) and os.path.isfile(os.path.join(PD_dir,"Y_Mapper_rv.txt")):
                        dist_Mapper_1.append(np.loadtxt(os.path.join(PD_dir,"Y_Mapper_norm_dim_1_wd.txt")))
                        dist_random_1.append(np.loadtxt(os.path.join(PD_dir,"Y_random_norm_dim_1_wd.txt")))
                        dist_Isomap_1.append(np.loadtxt(os.path.join(PD_dir,"Y_Isomap_norm_dim_1_wd.txt")))
                    
                        rv_Mapper.append(np.loadtxt(os.path.join(PD_dir,"Y_Mapper_rv.txt")))
                        rv_random.append(np.loadtxt(os.path.join(PD_dir,"Y_random_rv.txt")))
                        rv_Isomap.append(np.loadtxt(os.path.join(PD_dir,"Y_Isomap_rv.txt")))
              
           
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(211)

                labels=["Mapper-based LIsomap","Random LIsomap", "Isomap"]
                x = np.arange(0, 3, 1)
                
                for i in range (0, len(dist_Mapper_1)):
                    y_data = [dist_Mapper_1[i], dist_random_1[i],  dist_Isomap_1[i]]
                    plt.plot(x, y_data)
                plt.xticks(x, labels)


                ax = fig.add_subplot(212)
                
                for i in range (0, len(dist_Mapper_1)):
                    y_data = [rv_Mapper[i], rv_random[i], rv_Isomap[i]]
                    plt.plot(x, y_data)
                plt.xticks(x, labels)
                plt.show()


def _evaluation(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    for dataset in _DATA_TYPE:
        print ("------------------"+dataset+"--------------------------")        
        for noise in noise_:
            for outlier in outlier_:
                data_dir_ = os.path.join(_DATA_DIR, dataset+'_'+str(noise)+'_'+str(outlier))
                for exp in range (1,exps+1):
                    print ("----------Exp_"+str(exp)+"----------------")

                    print ('\nSTEP 1: generate persistent diagram')
                    data_dir = os.path.join(data_dir_, 'Exp_'+str(exp))
                    Json_dir = os.path.join(data_dir, 'Json')
                    X_norm = os.path.join(data_dir, "X_norm.txt")
                    Projection_dir = os.path.join(data_dir, 'Projection')
                    PD_dir = os.path.join(data_dir, 'PD')
                    file_util.make_dir(PD_dir)
                    PD_file = os.path.join(PD_dir,"X_norm_PD.txt")
                    PD = subprocess.check_output('ripser --format point-cloud --dim 1 '+ X_norm, shell=True)
                    file_X_norm = open(PD_file, 'w')
                    file_X_norm.write(PD)
                    file_X_norm.close()
                

                    # Generate persistence diagrams for projected Data
                    for Y_file in ["Y_Mapper", "Y_random", "Y_Isomap"]:
                        Y_norm = os.path.join(Projection_dir, Y_file+".txt")
                        PD_file = os.path.join(PD_dir,Y_file+"_norm_PD.txt")
                        PD = subprocess.check_output('ripser --format point-cloud --dim 1 '+ Y_norm, shell=True)
                        file_Y = open(PD_file, 'w')
                        file_Y.write(PD)
                        file_Y.close()
                    # Extract dim-1 information in persistence diagrams
                    for pd_file in ["X", "Y_Mapper", "Y_random" ,"Y_Isomap"]:
                        PD_file = os.path.join(PD_dir, pd_file+"_norm_PD.txt")
                        PD_1_file = os.path.join(PD_dir, pd_file+"_norm_dim_1.txt")
		        PD_0_file = os.path.join(PD_dir, pd_file+"_norm_dim_0.txt")

                        dataStart = False
                        with open(PD_file) as f:
                            for line in f:
                                if(dataStart):
                                    new = line[2:-2].split(',')
                                    new = ' '.join(new)
                                    file_X_norm = open(PD_1_file, 'a')
                                    file_X_norm.write(new+'\n')
                                    file_X_norm.close()
                                if (line == 'persistence intervals in dim 1:\n'):
                                    dataStart = True

                        dataStart = False
                        with open(PD_file) as f:
                            for line in f:
                                if (line == ' [0, )\n'):
                                    dataStart = False
                                if(dataStart):
                                    new = line[2:-2].split(',')
                                    new = ' '.join(new)
                                    file_Y = open(PD_0_file, 'a')
                                    file_Y.write(new+'\n')
                                    file_Y.close()
                                if (line == 'persistence intervals in dim 0:\n'):
                                    dataStart = True


                    print ('\nSTEP 2: Compute dim-1 persistent distance')   
                    X_dim_0 = os.path.join(PD_dir, "X_norm_dim_0.txt")
                    X_dim_1 = os.path.join(PD_dir, "X_norm_dim_1.txt")
                    
                    # Compute pairwise wessestain distance with p =2
                    for Y_file_ in ["Y_Mapper", "Y_random", "Y_Isomap"]:
                        Y_file = os.path.join(PD_dir, Y_file_+"_norm_dim_1.txt")                
                        XY_w = subprocess.check_output('wasserstein_dist ' +  X_dim_1 + ' ' +Y_file+ ' 2', shell=True).replace('\n', '')

                        WD_OutFile =os.path.join(PD_dir, Y_file_+ "_norm_dim_1_wd.txt")
                        file_Y = open(WD_OutFile, 'w')
                        ele = XY_w
                        file_Y.write(ele)
                        file_Y.close()
                            
                        Y_file = os.path.join(PD_dir, Y_file_+"_norm_dim_0.txt")                
                        XY_w = subprocess.check_output('wasserstein_dist ' +  X_dim_0 + ' ' +Y_file+ ' 2', shell=True).replace('\n', '')
                        
                        WD_OutFile =os.path.join(PD_dir, Y_file_+ "_norm_dim_0_wd.txt")
                        file_Y = open(WD_OutFile, 'w')
                        ele = XY_w
                        file_Y.write(ele)
                        file_Y.close()
     
                    print ("compute residual variance")
                    X = np.loadtxt(X_norm)
                    n_neighbors=8
                    n_components=2
                    eigen_solver='auto'
                    n_jobs=1
                    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
                    nbrs_.fit(X)
                    kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)
    
                    X_dist = graph_shortest_path(kng,
                                                 method='auto',
                                                 directed=False)
                    X_list = np.reshape(X_dist, len(X_dist)*len(X_dist))
                    print ("X_dist Done!")

                    for Y_file_ in ["Y_Mapper", "Y_random", "Y_Isomap"]:
                        proj_ = np.loadtxt(os.path.join(Projection_dir, Y_file_+".txt"))
                        proj_dist = euclidean_distances(proj_, proj_)
                        proj_list = np.reshape(proj_dist, len(proj_dist)*len(proj_dist))

                        pcc, pvalue = pearsonr(X_list, proj_list)
                        rv = 1-pcc**2
                        rv_file = os.path.join(PD_dir, Y_file_+"_rv.txt")
                        np.savetxt(rv_file, rv.reshape(1,), fmt="%f")

def _plot_evaluation_R(_DATA_TYPE):
    for dataset in _DATA_TYPE:
        print ("------------------"+dataset+"--------------------------")      
        data_dir = os.path.join(_DATA_DIR, dataset)
    
        dist_Mapper_1 = []
        dist_random_1 = []
        dist_Isomap_1 = []
        
        rv_Mapper = []
        rv_random = []
        rv_Isomap = []
         
            
        PD_dir = os.path.join(data_dir, 'PD')
                    
     	if os.path.isfile(os.path.join(PD_dir,"Y_Mapper_norm_dim_1_wd.txt")) and os.path.isfile(os.path.join(PD_dir,"Y_Mapper_rv.txt")):
            dist_Mapper_1.append(np.loadtxt(os.path.join(PD_dir,"Y_Mapper_norm_dim_1_wd.txt")))
            dist_random_1.append(np.loadtxt(os.path.join(PD_dir,"Y_random_norm_dim_1_wd.txt")))
            dist_Isomap_1.append(np.loadtxt(os.path.join(PD_dir,"Y_Isomap_norm_dim_1_wd.txt")))
                    
            rv_Mapper.append(np.loadtxt(os.path.join(PD_dir,"Y_Mapper_rv.txt")))
            rv_random.append(np.loadtxt(os.path.join(PD_dir,"Y_random_rv.txt")))
            rv_Isomap.append(np.loadtxt(os.path.join(PD_dir,"Y_Isomap_rv.txt")))
              
           
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)

        labels=["Mapper-based LIsomap","Random LIsomap", "Isomap"]
        x = np.arange(0, 3, 1)
                
        for i in range (0, len(dist_Mapper_1)):
            y_data = [dist_Mapper_1[i], dist_random_1[i],  dist_Isomap_1[i]]
            plt.plot(x, y_data)
        plt.xticks(x, labels)


        ax = fig.add_subplot(212)
                
        for i in range (0, len(dist_Mapper_1)):
            y_data = [rv_Mapper[i], rv_random[i], rv_Isomap[i]]
            plt.plot(x, y_data)
        plt.xticks(x, labels)
        plt.show()


def _evaluation_R(_DATA_TYPE):
    for dataset in _DATA_TYPE:
        print ("------------------"+dataset+"--------------------------")        
        print ('\nSTEP 1: generate persistent diagram')
        data_dir = os.path.join(_DATA_DIR, dataset)
        Json_dir = os.path.join(data_dir, 'Json')
        Projection_dir = os.path.join(data_dir, 'Projection')

        infile = os.path.join(data_dir, 'X.txt')
        X = np.loadtxt(infile)
        outfile = os.path.join(data_dir, 'X_norm.txt')
        X_norm = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        np.savetxt(outfile, X_norm, fmt ="%f")
        X_norm = os.path.join(data_dir, "X_norm.txt")

        PD_dir = os.path.join(data_dir, 'PD')
        file_util.make_dir(PD_dir)
        PD_file = os.path.join(PD_dir,"X_norm_PD.txt")
        PD = subprocess.check_output('ripser --format point-cloud --dim 1 '+ X_norm, shell=True)
        file_X_norm = open(PD_file, 'w')
        file_X_norm.write(PD)
        file_X_norm.close()
                

        # Generate persistence diagrams for projected Data
        for Y_file in ["Y_Mapper", "Y_random", "Y_Isomap"]:
            Y_norm = os.path.join(Projection_dir, Y_file+".txt")
            PD_file = os.path.join(PD_dir,Y_file+"_norm_PD.txt")
            PD = subprocess.check_output('ripser --format point-cloud --dim 1 '+ Y_norm, shell=True)
            file_Y = open(PD_file, 'w')
            file_Y.write(PD)
            file_Y.close()
        # Extract dim-1 information in persistence diagrams
        for pd_file in ["X", "Y_Mapper", "Y_random" ,"Y_Isomap"]:
            PD_file = os.path.join(PD_dir, pd_file+"_norm_PD.txt")
            PD_1_file = os.path.join(PD_dir, pd_file+"_norm_dim_1.txt")
	    PD_0_file = os.path.join(PD_dir, pd_file+"_norm_dim_0.txt")
            
            dataStart = False
            with open(PD_file) as f:
                for line in f:
                    if(dataStart):
                        new = line[2:-2].split(',')
                        new = ' '.join(new)
                        file_X_norm = open(PD_1_file, 'a')
                        file_X_norm.write(new+'\n')
                        file_X_norm.close()
                    if (line == 'persistence intervals in dim 1:\n'):
                        dataStart = True

            dataStart = False
            with open(PD_file) as f:
                for line in f:
                    if (line == ' [0, )\n'):
                        dataStart = False
                    if(dataStart):
                        new = line[2:-2].split(',')
                        new = ' '.join(new)
                        file_Y = open(PD_0_file, 'a')
                        file_Y.write(new+'\n')
                        file_Y.close()
                    if (line == 'persistence intervals in dim 0:\n'):
                        dataStart = True


        print ('\nSTEP 2: Compute dim-1 persistent distance')   
        X_dim_0 = os.path.join(PD_dir, "X_norm_dim_0.txt")
        X_dim_1 = os.path.join(PD_dir, "X_norm_dim_1.txt")
                    
        # Compute pairwise wessestain distance with p =2
        for Y_file_ in ["Y_Mapper", "Y_random", "Y_Isomap"]:
            Y_file = os.path.join(PD_dir, Y_file_+"_norm_dim_1.txt")                
            XY_w = subprocess.check_output('wasserstein_dist ' +  X_dim_1 + ' ' +Y_file+ ' 2', shell=True).replace('\n', '')
            
            WD_OutFile =os.path.join(PD_dir, Y_file_+ "_norm_dim_1_wd.txt")
            file_Y = open(WD_OutFile, 'w')
            ele = XY_w
            file_Y.write(ele)
            file_Y.close()
            
            Y_file = os.path.join(PD_dir, Y_file_+"_norm_dim_0.txt")                
            XY_w = subprocess.check_output('wasserstein_dist ' +  X_dim_0 + ' ' +Y_file+ ' 2', shell=True).replace('\n', '')
            
            WD_OutFile =os.path.join(PD_dir, Y_file_+ "_norm_dim_0_wd.txt")
            file_Y = open(WD_OutFile, 'w')
            ele = XY_w
            file_Y.write(ele)
            file_Y.close()
     
        print ("compute residual variance")
        X = np.loadtxt(X_norm)
        n_neighbors=8
        n_components=2
        eigen_solver='auto'
        n_jobs=1
        nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
        nbrs_.fit(X)
        kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)
        
        X_dist = graph_shortest_path(kng,
                                     method='auto',
                                     directed=False)
        X_list = np.reshape(X_dist, len(X_dist)*len(X_dist))
        print ("X_dist Done!")

        for Y_file_ in ["Y_Mapper", "Y_random", "Y_Isomap"]:
            proj_ = np.loadtxt(os.path.join(Projection_dir, Y_file_+".txt"))
            proj_dist = euclidean_distances(proj_, proj_)
            proj_list = np.reshape(proj_dist, len(proj_dist)*len(proj_dist))
            
            pcc, pvalue = pearsonr(X_list, proj_list)
            rv = 1-pcc**2
            rv_file = os.path.join(PD_dir, Y_file_+"_rv.txt")
            np.savetxt(rv_file, rv.reshape(1,), fmt="%f")
            
            
 
