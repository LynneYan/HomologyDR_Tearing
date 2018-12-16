import numpy as np
import os
import km
import file_util
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import euclidean
import sklearn
import json

_DATA_DIR = '../Data'

def _compute_skeleton(jsonfile, X):
    Landmark=[]
    with open(jsonfile) as data_file:    
        data = json.load(data_file)

    node_num = len(data['nodes'])                    
    for i in range (0, node_num):    
        children_num = len(data['nodes'][i]['children'])
        children = []
        for j in range (0, children_num):
            children.append(X[int(data['nodes'][i]['children'][j]['name'])])
            #color[int(data['nodes'][i]['children'][j]['name'])] = i
        children = np.array(children)
        kmeans = KMeans(n_clusters=1).fit(children)
        ck_km = kmeans.cluster_centers_[0]
        dist_min = 10000000
        for i in range (0, children_num):
            if euclidean(children[i], ck_km)<dist_min:
                dist_min = euclidean(children[i], ck_km)
                center = children[i]
        Landmark.append(center)
    Landmark = np.array(Landmark)
    links = []
    link_num = len(data['links'])
    for k in range (0, link_num):
        links.append((int(data['links'][k]['source']),int(data['links'][k]['target'])))
    return Landmark, links

def _plot_data_with_skeleton(X, links, color, Landmark, filter_function, exp, data_dir, Json_dir, BP_file=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, c=color,cmap=plt.cm.get_cmap('coolwarm'), alpha=0.7)
    ax.scatter(Landmark[:, 0], Landmark[:, 1], Landmark[:, 2], cmap=plt.cm.Spectral, c='black', s=100,  marker=(5, 1),label='Landmarks(Centroid)')
                        
    if filter_function == "base_point_geodesic_distance" and os.path.isfile(os.path.join(Json_dir,"BP.txt")):
	base_point = np.loadtxt(BP_file)
        ax.scatter(base_point[0], base_point[1], base_point[2], cmap=plt.cm.Spectral, c='red', s=250,  marker=(5, 1),label='Base Point')
    for i in range (0, len(links)):
        ax.plot( [Landmark[links[i][0]][0], Landmark[links[i][1]][0]], [Landmark[links[i][0]][1], Landmark[links[i][1]][1]], [Landmark[links[i][0]][2], Landmark[links[i][1]][2]],color = 'black')
    ax.legend()
    title = 'Filter Function: '+filter_function+'; Landmark size: '+str(len(Landmark))
    ax.set_title(title)
    Plot_dir = os.path.join(data_dir, 'Plot')
    file_util.make_dir(Plot_dir)
    if exp:
        Plot_dir = os.path.join(Plot_dir, 'Exp_'+str(exp)+"_"+filter_function+".png")
    else:
        Plot_dir = os.path.join(Plot_dir, filter_function+".png")
    plt.axis("equal")
    plt.show()
    fig.savefig(Plot_dir)


def _generate_Json_files(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    for dataset in _DATA_TYPE:
        print "------------------"+dataset+"--------------------------"
        for noise in noise_:
            for outlier in outlier_:
                data_dir_ = os.path.join(_DATA_DIR, dataset+'_'+str(noise)+'_'+str(outlier))
                for exp in range (1, exps+1):
                    print "----------Exp_"+str(exp)+"----------------"
                    data_dir = os.path.join(data_dir_, 'Exp_'+str(exp))
                    Json_dir = os.path.join(data_dir, 'Json')
                    file_util.make_dir(Json_dir)
                    X, color = file_util._get_normalized_input_data(data_dir)

                    
                    #Filter functions: "density_estimator","integral_geodesic_distance",
                    #"height","width","Guass_density_auto", "eccentricity","graph_Laplacian"
                    
                    for filter_function in ["base_point_geodesic_distance"]:
                        
                        jsonfile = os.path.join(Json_dir, 'Y_'+filter_function+'.json')
                        htmlfile = os.path.join(Json_dir, 'Y_'+filter_function+'.html')
                        colorfile  = os.path.join(Json_dir, 'Y_'+filter_function+'.txt') 
                        BP_file =  os.path.join(Json_dir, 'BP.txt')
                        dist_file =  os.path.join(Json_dir, 'X_dist.txt')

                        mapper = km.KeplerMapper()
                        if filter_function == "base_point_geodesic_distance":
                            if outlier == 0:
                                lens = mapper.fit_transform(X, projection="base_point_geodesic_distance", color_file=colorfile, BP_file = BP_file, dist_file = dist_file)
                            else:
                                lens = mapper.fit_transform(X, projection="base_point_geodesic_distance_outlier", color_file=colorfile, BP_file = BP_file)
                        else:
                            lens = mapper.fit_transform(X, projection=filter_function, color_file=colorfile)
		
                        if dataset == 'Swiss_hole':
                            if outlier == 0:
                                eps = 0.6
                                min_sample = 5
                                nr_cubes = 20
                                overlap_perc = 0.2
                            else:
                                eps = 0.25
                                min_sample = 5
                                nr_cubes = 30
                                overlap_perc = 0.2

                        if dataset == 'Fishing_net':
                            if outlier == 0:
                                eps = 0.4
                                min_sample = 5
                                nr_cubes = 30
                                overlap_perc = 0.5
                            else:
                                eps = 0.1
                                min_sample = 5
                                nr_cubes = 40
                                overlap_perc = 0.2
                            
                
                        graph = mapper.map(lens,X,
                                           clusterer = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_sample),
                                           nr_cubes = nr_cubes,
                                           overlap_perc = overlap_perc)
                        
                        mapper.visualize(graph, path_html=str(htmlfile))
                        mapper.save_json_file(graph,path_html=str(jsonfile))
                    
                        color = np.loadtxt(colorfile)
                        Landmark, links = _compute_skeleton(jsonfile, X)
                        _plot_data_with_skeleton(X, links, color, Landmark, filter_function, exp, data_dir,Json_dir, BP_file)

def _generate_Json_files_R(_DATA_TYPE):
    for dataset in _DATA_TYPE:
        print "------------------"+dataset+"--------------------------"
        data_dir = os.path.join(_DATA_DIR, dataset)
        Json_dir = os.path.join(data_dir, 'Json')
        file_util.make_dir(Json_dir)
        infile = os.path.join(data_dir, 'X.txt')
        X = np.loadtxt(infile)

        if dataset == 'octa':
            filter_function = "eccentricity"
            eps = 150
            min_sample = 5
            nr_cubes = 6
            overlap_perc = 0.2
                      
        if dataset == 'airfoil1':
            filter_function = "base_point_geodesic_distance"
            eps = 1.5
            min_sample = 3
            nr_cubes = 20
            overlap_perc = 0.2

        if dataset.startswith("cylinder"):
            filter_function = "width"
            eps = 0.4
            min_sample = 5
            nr_cubes = 20
            overlap_perc = 0.4
            
                        
        jsonfile = os.path.join(Json_dir, 'Y_'+filter_function+'.json')
        htmlfile = os.path.join(Json_dir, 'Y_'+filter_function+'.html')
        colorfile  = os.path.join(Json_dir, 'Y_'+filter_function+'.txt') 
        BP_file =  os.path.join(Json_dir, 'BP.txt')
        dist_file =  os.path.join(Json_dir, 'X_dist.txt')

        mapper = km.KeplerMapper()
        if filter_function == "base_point_geodesic_distance":
            lens = mapper.fit_transform(X, projection="base_point_geodesic_distance", color_file=colorfile, BP_file = BP_file, dist_file = dist_file)
        else:
            lens = mapper.fit_transform(X, projection=filter_function, color_file=colorfile)
               
        graph = mapper.map(lens,X,
                           clusterer = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_sample),
                           nr_cubes = nr_cubes,
                           overlap_perc = overlap_perc)
                        
        mapper.visualize(graph, path_html=str(htmlfile))
        mapper.save_json_file(graph,path_html=str(jsonfile))
                    
        color = np.loadtxt(colorfile)
        Landmark, links = _compute_skeleton(jsonfile, X)
        _plot_data_with_skeleton(X, links, color, Landmark, filter_function, None, data_dir,Json_dir, BP_file)
