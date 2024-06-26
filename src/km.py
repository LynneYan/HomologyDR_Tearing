from __future__ import division
import numpy as np
from collections import defaultdict
import json
import itertools
from sklearn import cluster, preprocessing, manifold, decomposition
from scipy.spatial import distance
from datetime import datetime
import sys
import inspect
import json
import random
from scipy import stats
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, euclidean
import math
import scipy.sparse
import scipy.sparse.linalg as spla
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import squareform, cdist, pdist
if sys.hexversion < 0x03000000:
    from itertools import izip as zip
    range = xrange

#from mapper import n_obs, cmappertoolserror
import progressreporter



class KeplerMapper(object):

  def __init__(self, verbose=2):
    self.verbose = verbose
    self.chunk_dist = []
    self.overlap_dist = []
    self.d = []
    self.nr_cubes = 0
    self.overlap_perc = 0
    self.clusterer = False

  def fit_transform(self, X, projection="sum", color_file=None, BP_file = None, dist_file = None, scaler=preprocessing.MinMaxScaler(), distance_matrix=False):
 
    self.inverse = X
    self.scaler = scaler
    self.projection = str(projection)
    self.distance_matrix = distance_matrix
    

    if self.distance_matrix in ["braycurtis", 
                           "canberra", 
                           "chebyshev", 
                           "cityblock", 
                           "correlation", 
                           "cosine", 
                           "dice", 
                           "euclidean", 
                           "hamming", 
                           "jaccard", 
                           "kulsinski", 
                           "mahalanobis", 
                           "matching", 
                           "minkowski", 
                           "rogerstanimoto", 
                           "russellrao", 
                           "seuclidean", 
                           "sokalmichener", 
                           "sokalsneath", 
                           "sqeuclidean", 
                           "yule"]:
      X = distance.squareform(distance.pdist(X, metric=distance_matrix))
      if self.verbose > 0:
        print("Created distance matrix, shape: %s, with distance metric `%s`"%(X.shape, distance_matrix))

    # Detect if projection is a class (for scikit-learn)
    try:
      p = projection.get_params()
      reducer = projection
      if self.verbose > 0:
        try:    
          projection.set_params(**{"verbose":self.verbose})
        except:
          pass
        print("\n..Projecting data using: \n\t%s\n"%str(projection))
      X = reducer.fit_transform(X)
    except:
      pass
    
    # Detect if projection is a string (for standard functions)
    if isinstance(projection, str):
      if self.verbose > 0:
        print("\n..Projecting data using: %s"%(projection))
      # Stats lenses
      if projection == "sum": # sum of row
        X = np.sum(X, axis=1).reshape((X.shape[0],1))
      if projection == "mean": # mean of row
        X = np.mean(X, axis=1).reshape((X.shape[0],1))
      if projection == "median": # mean of row
        X = np.median(X, axis=1).reshape((X.shape[0],1))
      if projection == "max": # max of row
        X = np.max(X, axis=1).reshape((X.shape[0],1))
      if projection == "min": # min of row
        X = np.min(X, axis=1).reshape((X.shape[0],1))
      if projection == "std": # std of row
        X = np.std(X, axis=1).reshape((X.shape[0],1))
      if projection == "l2norm":
        X = np.linalg.norm(X, axis=1).reshape((X.shape[0], 1))
      if projection == "height": # std of row
        height = X[:, 2].reshape((X.shape[0],1))
        X =height   
      if projection == "width": # std of row
        height = X[:, 0].reshape((X.shape[0],1))
        X =height   
      if projection == "base_point_geodesic_distance":
        X = Base_Point_Geodesic_Distance(X, 8, BP_file, dist_file).reshape((X.shape[0],1))

      if projection == "base_point_geodesic_distance_outlier":
        X = Base_Point_Geodesic_Distance_outlier(X, 8, BP_file).reshape((X.shape[0],1))

      if projection == "dist_mean": # Distance of x to mean of X
        X_mean = np.mean(X, axis=0) 
        X = np.sum(np.sqrt((X - X_mean)**2), axis=1).reshape((X.shape[0],1))
        
      if projection == "eccentricity":
          X = eccentricity(X, 1, {}, None).reshape((X.shape[0],1))
      if projection == "Guass_density":
          #sigma=float(raw_input("Please enter sigma: "))
          sigma=0.8
          X = Gauss_density(X, sigma, {}, None).reshape((X.shape[0],1))

      if projection == "density_estimator":
	  k = 15
          X = eccentricity(X, 15).reshape((X.shape[0],1))

      if projection == "integral_geodesic_distance":
          X = Integral_Geodesic_Distance(X, 10).reshape((X.shape[0],1))

      if projection == "graph_Laplacian":
          #eps = float(raw_input("Please enter espion: "))
	  eps = 0.2
          X = eccentricity(X, eps).reshape((X.shape[0],1))

      if projection == "Guass_density_auto":
  	  kde = stats.gaussian_kde(X.T)
  	  X = kde(X.T).reshape((X.shape[0],1))    

      if "knn_distance_" in projection:
        n_neighbors = int(projection.split("_")[2])
        if self.distance_matrix: # We use the distance matrix for finding neighbors
          X = np.sum(np.sort(X, axis=1)[:,:n_neighbors], axis=1).reshape((X.shape[0], 1))
        else:
          from sklearn import neighbors
          nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
          nn.fit(X)
          X = np.sum(nn.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)[0], axis=1).reshape((X.shape[0], 1))

    np.savetxt(color_file, X, fmt='%f')
    # Detect if projection is a list (with dimension indices)
    if isinstance(projection, list):
      if self.verbose > 0:
        print("\n..Projecting data using: %s"%(str(projection)))
      X = X[:,np.array(projection)]
      
    # Scaling
    if scaler is not None:
      if self.verbose > 0:
        print("\n..Scaling with: %s\n"%str(scaler))
      X = scaler.fit_transform(X)

    return X

  def map(self, projected_X, inverse_X=None, clusterer=cluster.DBSCAN(eps=0.5,min_samples=3), nr_cubes=10, overlap_perc=0.1):
    # This maps the data to a simplicial complex. Returns a dictionary with nodes and links.
    # 
    # Input:    projected_X. A Numpy array with the projection/lens. 
    # Output:    complex. A dictionary with "nodes", "links" and "meta information"
    #
    # parameters
    # ----------
    # projected_X  	projected_X. A Numpy array with the projection/lens. Required.
    # inverse_X    	Numpy array or None. If None then the projection itself is used for clustering.
    # clusterer    	Scikit-learn API compatible clustering algorithm. Default: DBSCAN
    # nr_cubes    	Int. The number of intervals/hypercubes to create.
    # overlap_perc  Float. The percentage of overlap "between" the intervals/hypercubes.
    
    start = datetime.now()
    
    # Helper function
    def cube_coordinates_all(nr_cubes, nr_dimensions):
      # Helper function to get origin coordinates for our intervals/hypercubes
      # Useful for looping no matter the number of cubes or dimensions
      # Example:   	if there are 4 cubes per dimension and 3 dimensions 
      #       		return the bottom left (origin) coordinates of 64 hypercubes, 
      #       		as a sorted list of Numpy arrays
      # TODO: elegance-ify...
      l = []
      for x in range(nr_cubes):
        l += [x] * nr_dimensions
      return [np.array(list(f)) for f in sorted(set(itertools.permutations(l,nr_dimensions)))]
    
    nodes = defaultdict(list)
    links = defaultdict(list)
    meta = defaultdict(list)
    graph = {}
    self.nr_cubes = nr_cubes
    self.clusterer = clusterer
    self.overlap_perc = overlap_perc
    
    # If inverse image is not provided, we use the projection as the inverse image (suffer projection loss)
    if inverse_X is None:
      inverse_X = projected_X

    if self.verbose > 0:
      print("Mapping on data shaped %s using lens shaped %s\n"%(str(inverse_X.shape), str(projected_X.shape)))  

    # We chop up the min-max column ranges into 'nr_cubes' parts
    self.chunk_dist = (np.max(projected_X, axis=0) - np.min(projected_X, axis=0))/nr_cubes

    # We calculate the overlapping windows distance 
    self.overlap_dist = self.overlap_perc * self.chunk_dist

    # We find our starting point
    self.d = np.min(projected_X, axis=0)
    
    # Use a dimension index array on the projected X
    # (For now this uses the entire dimensionality, but we keep for experimentation)
    di = np.array([x for x in range(projected_X.shape[1])])
    
    # Prefix'ing the data with ID's
    ids = np.array([x for x in range(projected_X.shape[0])])
    projected_X = np.c_[ids,projected_X]
    inverse_X = np.c_[ids,inverse_X]

    # Algo's like K-Means, have a set number of clusters. We need this number
    # to adjust for the minimal number of samples inside an interval before
    # we consider clustering or skipping it.
    cluster_params = self.clusterer.get_params()
    try:
      min_cluster_samples = cluster_params["n_clusters"]
    except:
      min_cluster_samples = 1
    if self.verbose > 0:
      print("Minimal points in hypercube before clustering: %d"%(min_cluster_samples))

    # Subdivide the projected data X in intervals/hypercubes with overlap
    if self.verbose > 0:
      total_cubes = len(list(cube_coordinates_all(nr_cubes,di.shape[0])))
      print("Creating %s hypercubes."%total_cubes)

    for i, coor in enumerate(cube_coordinates_all(nr_cubes,di.shape[0])):

      # Slice the hypercube
      hypercube = projected_X[ np.invert(np.any((projected_X[:,di+1] >= self.d[di] + (coor * self.chunk_dist[di])) & 
          (projected_X[:,di+1] < self.d[di] + (coor * self.chunk_dist[di]) + self.chunk_dist[di] + self.overlap_dist[di]) == False, axis=1 )) ]
      
      if self.verbose > 1:
        print("There are %s points in cube_%s / %s with starting range %s"%
              (hypercube.shape[0],i,total_cubes,self.d[di] + (coor * self.chunk_dist[di])))
      
      # If at least min_cluster_samples samples inside the hypercube
      if hypercube.shape[0] >= min_cluster_samples:
        # Cluster the data point(s) in the cube, skipping the id-column
        # Note that we apply clustering on the inverse image (original data samples) that fall inside the cube.
        inverse_x = inverse_X[[int(nn) for nn in hypercube[:,0]]]
        
        clusterer.fit(inverse_x[:,1:])
        
        if self.verbose > 1:
          print("Found %s clusters in cube_%s\n"%(np.unique(clusterer.labels_[clusterer.labels_ > -1]).shape[0],i))
        
        #Now for every (sample id in cube, predicted cluster label)
        for a in np.c_[hypercube[:,0],clusterer.labels_]:
          if a[1] != -1: #if not predicted as noise
            cluster_id = str(coor[0])+"_"+str(i)+"_"+str(a[1])+"_"+str(coor)+"_"+str(self.d[di] + (coor * self.chunk_dist[di])) # TODO: de-rudimentary-ify

            nodes[cluster_id].append( int(a[0]) ) # Append the member id's as integers
            meta[cluster_id] = {"size": hypercube.shape[0], "coordinates": coor}
            size=hypercube.shape[0]

      else:
        if self.verbose > 1:
          print("Cube_%s is empty.\n"%(i))

    # Create links when clusters from different hypercubes have members with the same sample id.
    candidates = itertools.combinations(nodes.keys(),2)
    for candidate in candidates:
      # if there are non-unique members in the union
      if len(nodes[candidate[0]]+nodes[candidate[1]]) != len(set(nodes[candidate[0]]+nodes[candidate[1]])):
        links[candidate[0]].append( candidate[1] )

    # Reporting
    if self.verbose > 0:
      nr_links = 0
      for k in links:
        nr_links += len(links[k])
      print("\ncreated %s edges and %s nodes in %s."%(nr_links,len(nodes),str(datetime.now()-start)))
    graph["nodes"] = nodes
    graph["links"] = links
    graph["meta_graph"] = self.projection
    graph["meta_nodes"] = meta
    return graph

  def visualize(self, complex, color_function="", path_html="mapper_visualization_output.html", title="My Data", 
          graph_link_distance=30, graph_gravity=0.1, graph_charge=-120, custom_tooltips=None, width_html=0, 
          height_html=0, show_tooltips=True, show_title=True, show_meta=True):
    # Turns the dictionary 'complex' in a html file with d3.js
    #
    # Input:      complex. Dictionary (output from calling .map())
    # Output:      a HTML page saved as a file in 'path_html'.
    # 
    # parameters
    # ----------
    # color_function    	string. Not fully implemented. Default: "" (distance to origin)
    # path_html        		file path as string. Where to save the HTML page.
    # title          		string. HTML page document title and first heading.
    # graph_link_distance  	int. Edge length.
    # graph_gravity     	float. "Gravity" to center of layout.
    # graph_charge      	int. charge between nodes.
    # custom_tooltips   	None or Numpy Array. You could use "y"-label array for this.
    # width_html        	int. Width of canvas. Default: 0 (full width)
    # height_html       	int. Height of canvas. Default: 0 (full height)
    # show_tooltips     	bool. default:True
    # show_title      		bool. default:True
    # show_meta        		bool. default:True

    # Format JSON for D3 graph
    json_s = {}
    json_s["nodes"] = []
    json_s["links"] = []
    k2e = {} # a key to incremental int dict, used for id's when linking
    print(color_function)
    for e, k in enumerate(complex["nodes"]):
      # Tooltip and node color formatting, TODO: de-mess-ify
      if custom_tooltips is not None:
        autism_n=0
        control_n=0
        num=0
        for f in complex["nodes"][k]:
          num+=1
          if f<49:
            autism_n+=1
          else:
            control_n+=1
        color_n=int((((control_n-autism_n)/num-2)*10)/5)*5/10
        print(color_n)
        tooltip_s = "<h2>Cluster %s</h2> Contains %s members.<br>%s"%(k,len(complex["nodes"][k]), " ".join([str(f) for f in custom_tooltips[complex["nodes"][k]]]))
        if color_function == "average_signal_cluster":
          tooltip_i = int(((sum([f for f in custom_tooltips[complex["nodes"][k]]]) / len(custom_tooltips[complex["nodes"][k]])) * 30) )
          json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(tooltip_i)})
        elif color_function =="PC2":
          json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(complex["meta_nodes"][k]["coordinates"][1]*3)})
        elif color_function =="label":
          json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(color_n)})
        else:
          json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(complex["meta_nodes"][k]["coordinates"][0]*3)})
      else:
        tooltip_s = "<h2>Cluster %s</h2>Contains %s members."%(k,len(complex["nodes"][k]))
        json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(complex["meta_nodes"][k]["coordinates"][0])})
      k2e[k] = e
    for k in complex["links"]:
      for link in complex["links"][k]:
        json_s["links"].append({"source": k2e[k], "target":k2e[link],"value":1})

    # Width and height of graph in HTML output
    if width_html == 0:
      width_css = "100%"
      width_js = 'document.getElementById("holder").offsetWidth-20'
    else:
      width_css = "%spx" % width_html
      width_js = "%s" % width_html
    if height_html == 0:
      height_css = "100%"
      height_js = 'document.getElementById("holder").offsetHeight-20'
    else:
      height_css = "%spx" % height_html
      height_js = "%s" % height_html
    
    # Whether to show certain UI elements or not
    if show_tooltips == False:
      tooltips_display = "display: none;"
    else:
      tooltips_display = ""
      
    if show_meta == False:
      meta_display = "display: none;"
    else:
      meta_display = ""
      
    if show_title == False:
      title_display = "display: none;"
    else:
      title_display = ""  
    
    with open(path_html,"wb") as outfile:
      html = """<!DOCTYPE html>
    <meta charset="utf-8">
    <meta name="generator" content="KeplerMapper">
    <title>%s | KeplerMapper</title>
    <link href='https://fonts.googleapis.com/css?family=Roboto:700,300' rel='stylesheet' type='text/css'>
    <style>
    * {margin: 0; padding: 0;}
    html { height: 100%%;}
    body {background: #111; height: 100%%; font: 100 16px Roboto, Sans-serif;}
    .link { stroke: #999; stroke-opacity: .333;  }
    .divs div { border-radius: 50%%; background: red; position: absolute; }
    .divs { position: absolute; top: 0; left: 0; }
    #holder { position: relative; width: %s; height: %s; background: #111; display: block;}
    h1 { %s padding: 20px; color: #fafafa; text-shadow: 0px 1px #000,0px -1px #000; position: absolute; font: 300 30px Roboto, Sans-serif;}
    h2 { text-shadow: 0px 1px #000,0px -1px #000; font: 700 16px Roboto, Sans-serif;}
    .meta {  position: absolute; opacity: 0.9; width: 220px; top: 80px; left: 20px; display: block; %s background: #000; line-height: 25px; color: #fafafa; border: 20px solid #000; font: 100 16px Roboto, Sans-serif;}
    div.tooltip { position: absolute; width: 380px; display: block; %s padding: 20px; background: #000; border: 0px; border-radius: 3px; pointer-events: none; z-index: 999; color: #FAFAFA;}
    }
    </style>
    <body>
    <div id="holder">
      <h1>%s</h1>
      <p class="meta">
      <b>Lens</b><br>%s<br><br>
      <b>Cubes per dimension</b><br>%s<br><br>
      <b>Overlap percentage</b><br>%s%%<br><br>
      <b>Color Function</b><br>%s( %s )<br><br>
      <b>Clusterer</b><br>%s<br><br>
      <b>Scaler</b><br>%s
      </p>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
    <script>
    var width = %s,
      height = %s;
    var color = d3.scale.ordinal()
      .domain(["-3", "-2.5" ,"-2", "-1.5","-1","0","1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"])
      .range(["#41b6c4","#41b6c4","#253494","#c994c7","#ffffcc","#f768a1","#FF0000","#FF1400","#FF2800","#FF3c00","#FF5000","#FF6400","#FF7800","#FF8c00","#FFa000","#FFb400","#FFc800","#FFdc00","#FFf000","#fdff00","#b0ff00","#65ff00","#17ff00","#00ff36","#00ff83","#00ffd0","#00e4ff","#00c4ff","#00a4ff","#00a4ff","#0084ff","#0064ff","#0044ff","#0022ff","#0002ff","#0100ff","#0300ff","#0500ff"]);
    var force = d3.layout.force()
      .charge(%s)
      .linkDistance(%s)
      .gravity(%s)
      .size([width, height]);
    var svg = d3.select("#holder").append("svg")
      .attr("width", width)
      .attr("height", height);
    
    var div = d3.select("#holder").append("div")   
      .attr("class", "tooltip")               
      .style("opacity", 0.0);
    
    var divs = d3.select('#holder').append('div')
      .attr('class', 'divs')
      .attr('style', function(d) { return 'overflow: hidden; width: ' + width + 'px; height: ' + height + 'px;'; });  
    
      graph = %s;
      force
        .nodes(graph.nodes)
        .links(graph.links)
        .start();
      var link = svg.selectAll(".link")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link")
        .style("stroke-width", function(d) { return Math.sqrt(d.value); });
      var node = divs.selectAll('div')
      .data(graph.nodes)
        .enter().append('div')
        .on("mouseover", function(d) {      
          div.transition()        
            .duration(200)      
            .style("opacity", .9);
          div .html(d.tooltip + "<br/>")  
            .style("left", (d3.event.pageX + 100) + "px")     
            .style("top", (d3.event.pageY - 28) + "px");    
          })                  
        .on("mouseout", function(d) {       
          div.transition()        
            .duration(500)      
            .style("opacity", 0);   
        })
        .call(force.drag);
      
      node.append("title")
        .text(function(d) { return d.name; });
      force.on("tick", function() {
      link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });
      node.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; })
        .attr('style', function(d) { return 'width: ' + ((d.group+1) * 2) + 'px; height: ' + ((d.group+1) * 2) + 'px; ' + 'left: '+(d.x-(d.group+1))+'px; ' + 'top: '+(d.y-(d.group+1))+'px; background: '+color(d.color)+'; box-shadow: 0px 0px 3px #111; box-shadow: 0px 0px 33px '+color(d.color)+', inset 0px 0px 5px rgba(0, 0, 0, 0.2);'})
        ;
      });
    </script>"""%(title,width_css, height_css, title_display, meta_display, tooltips_display, title,complex["meta_graph"],self.nr_cubes,self.overlap_perc*100,color_function,complex["meta_graph"],str(self.clusterer),str(self.scaler),width_js,height_js,graph_charge,graph_link_distance,graph_gravity,json.dumps(json_s))
      outfile.write(html.encode("utf-8"))
    if self.verbose > 0:
      print("\nWrote d3.js graph to '%s'"%path_html)

  def save_json_file(self, complex, color_function="", path_html="data.json", title="My Data", 
          graph_link_distance=30, graph_gravity=0.1, graph_charge=-120, custom_tooltips=None, width_html=0, 
          height_html=0, show_tooltips=True, show_title=True, show_meta=True):
    json_s = {}
    json_s["links"] = []
    json_s["nodes"] = []
    k2e = {} # a key to incremental int dict, used for id's when linking
    for e, k in enumerate(complex["nodes"]):
      children = {}
      children["name"]=[]
      for ch in complex["nodes"][k]:
        children["name"].append({"name": ch})
      json_s["nodes"].append({"id": e, "group": e,"children": children["name"]})
      k2e[k] = e
    for k in complex["links"]:
      for link in complex["links"][k]:
        json_s["links"].append({"source": k2e[k], "target":k2e[link]})
    if self.verbose > 0:
      print("\nWrote d3.js graph to JSON File'%s'"%path_html)
    with open(path_html, 'w') as f:
      json.dump(json_s, f)


  def data_from_cluster_id(self, cluster_id, graph, data):
    if cluster_id in graph["nodes"]:
      cluster_members = graph["nodes"][cluster_id]
      cluster_members_data = data[cluster_members]
      return cluster_members_data
    else:
      return np.array([])

def eccentricity(data, exponent=1.,  metricpar={}, callback=None):
    if data.ndim==1:
        assert metricpar=={}, 'No optional parameter is allowed for a dissimilarity matrix.'
        ds = squareform(data, force='tomatrix')
        if exponent in (np.inf, 'Inf', 'inf'):
            return ds.max(axis=0)
        elif exponent==1.:
            ds = np.power(ds, exponent)
            return ds.sum(axis=0)/float(np.alen(ds))
        else:
            ds = np.power(ds, exponent)
            return np.power(ds.sum(axis=0)/float(np.alen(ds)), 1./exponent)
    else:
        progress = progressreporter.progressreporter(callback)
        N = np.alen(data)
        ecc = np.empty(N)
        if exponent in (np.inf, 'Inf', 'inf'):
            for i in range(N):
                ecc[i] = cdist(data[(i,),:], data, **metricpar).max()
                progress((i+1)*100//N)
        elif exponent==1.:
            for i in range(N):
                ecc[i] = cdist(data[(i,),:], data, **metricpar).sum()/float(N)
                progress((i+1)*100//N)
        else:
            for i in range(N):
                dsum = np.power(cdist(data[(i,),:], data, **metricpar),
                                exponent).sum()
                ecc[i] = np.power(dsum/float(N), 1./exponent)
                progress((i+1)*100//N)
        return ecc


def Density_Estimator(D, k):
    n_jobs = 1
    nbrs_ = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(D)
    kng = kneighbors_graph(nbrs_, k, mode='distance', n_jobs=n_jobs)

    DE = np.zeros(len(D))
    dist_matrix = kng.toarray() 
    for i in range (0, len(D)):
        for j in range (0, len(D)):
            if dist_matrix[i][j] > 0:
                DE[i] += dist_matrix[i][j]**2
    for i in range (0, len(D)):
        DE[i] = -1/k*math.sqrt(DE[i])

    return DE




def Base_Point_Geodesic_Distance_outlier(D, n_neighbors,BP_file):
    n_jobs = 1
    ALL_matrix = D
    n_neighbors =8
    kde = stats.gaussian_kde(D.T)
    KDE = kde(D.T).reshape((D.shape[0],1))
    Index = np.argmax(KDE)
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(ALL_matrix)
    kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)
    
    dist_matrix_ = graph_shortest_path(kng,method='auto', directed=False)
    G = dist_matrix_
    BPD = G[Index]
    """
    for i in range (0, len(BPD)):
        BPD[i]=math.log1p(BPD[i]/KDE[i])
    """
    np.savetxt(BP_file, D[Index], fmt="%f")
    return BPD

def Base_Point_Geodesic_Distance(D, n_neighbors,BP_file, dist_file):
    n_jobs = 1
    ALL_matrix = D
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(ALL_matrix)
    kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)
    dist_matrix_ = graph_shortest_path(kng,method='auto', directed=False)
    G = dist_matrix_
    np.savetxt(dist_file, G, fmt="%f")
     
    """
    x_max = np.max(D[:, 2])
    base_points = []
    for i in range (0, len(D)):
        if D[i,2]==x_max:
            base_points.append(D[i])
    base_points = np.array(base_points)
    kmeans = KMeans(n_clusters=1).fit(base_points)
    ck_km = kmeans.cluster_centers_[0]
    dist_min = 10
    for i in range (0, len(base_points)):
        if euclidean(base_points[i], ck_km)<dist_min:
            dist_min = euclidean(base_points[i], ck_km)
            base_point = base_points[i]
    
    for i in range(0, len(D)):
        if D[i, 0] == base_point[0] and D[i, 1] == base_point[1] and D[i, 2] == base_point[2]:
            Index = i
    """
    """
    x_max = np.max(D[:, 0])-0.1*np.max(D[:, 0])
    max_z = -10000
    Index = 0
    for i in range (0, len(D)):
        if D[i,0]>x_max:
            if D[i, 2]> max_z:
		max_z = D[i, 2]
		Index = i
    print Index

    Index = 0
    z_median = np.median(D[:, 0])
    for i in range (0, len(D)):
        if D[i, 0] == z_median:
             Index = i
    """
    """
    dist_min = 100000000
    Index = 0
    #base_point = np.ones(3)*0
    base_point = [0,0,-3]
    for i in range(0, len(D)):
        if euclidean(D[i], base_point)<dist_min:
             dist_min = euclidean(D[i], base_point)
             Index = i
   

    kde = stats.gaussian_kde(D.T)
    KDE = kde(D.T).reshape((D.shape[0],1))
    Index = np.argmax(KDE)
    """
    Index = np.argmax(G[random.randint(0, len(D)+1)])
 
    np.savetxt(BP_file, D[Index], fmt="%f")
    return G[Index]


def Integral_Geodesic_Distance(D, n_neighbors):
    n_jobs = 1
    ALL_matrix = D
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(ALL_matrix)
    kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)
    dist_matrix_ = graph_shortest_path(kng,method='auto', directed=False)
    G = dist_matrix_
    IGD = np.zeros(len(G))
    for i in range (0, len(G)):
        IGD[i] = sum(G[i])

    min_ = min(IGD)
    max_ = max(IGD)
    for i in range (0, len(G)):
        IGD[i] = (IGD[i]-min_)/max_
    return IGD


def Gauss_density(data, sigma, metricpar={}, callback=None):
    denom = -2.*sigma*sigma
    if data.ndim==1:
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        ds = squareform(data, force='tomatrix')
        dd = np.exp(ds*ds/denom)
        dens = dd.sum(axis=0)
    else:
        progress = progressreporter.progressreporter(callback)
        N = np.alen(data)
        dens = np.empty(N)
        for i in range(N):
            d = cdist(data[(i,),:], data, **metricpar)
            dens[i] = np.exp(d*d/denom).sum()
            progress(((i+1)*100//N))
        dens /= N*np.power(np.sqrt(2*np.pi)*sigma,data.shape[1])
    return dens

def graph_Laplacian(data, eps, n=1, k=1, weighted_edges=False, sigma_eps=1.,
                    normalized=True,
                    metricpar={}, verbose=True,
                    callback=None):
    assert n>=1, 'The rank of the eigenvector must be positive.'
    assert isinstance(k, int)
    assert k>=1
    if data.ndim==1:
        # dissimilarity matrix
        assert metricpar=={}, ('No optional parameter is allowed for a '
                               'dissimilarity matrix.')
        D = data
        N = n_obs(D)
    else:
        # vector data
        D = pdist(data, **metricpar)
        N = len(data)
    if callback:
        callback('Computing: neighborhood graph.')
    rowstart, targets, weights = \
        neighborhood_graph(D, k, eps, diagonal=True,
                           verbose=verbose, callback=callback)

    c = ncomp(rowstart, targets)
    if (c>1):
        print('The neighborhood graph has {0} components. Return zero values.'.
              format(c))
        return zero_filter(data)

    weights = Laplacian(rowstart, targets, weights, weighted_edges,
                        eps, sigma_eps, normalized)

    L = scipy.sparse.csr_matrix((weights, targets, rowstart))
    del weights, targets, rowstart

    if callback:
        callback('Computing: eigenvectors.')

    assert n<N, ('The rank of the eigenvector must be smaller than the number '
                 'of data points.')

    if hasattr(spla, 'eigsh'):
        w, v = spla.eigsh(L, k=n+1, which='SA')
    else: # for SciPy < 0.9.0
        w, v = spla.eigen_symmetric(L, k=n+1, which='SA')
    # Strange: computing more eigenvectors seems faster.
    #w, v = spla.eigsh(L, k=n+1, sigma=0., which='LM')
    if verbose:
        print('Eigenvalues: {0}.'.format(w))
    order = np.argsort(w)
    if w[order[0]]<0 and w[order[1]]<abs(w[order[0]]):
        raise RuntimeError('Negative eigenvalue of the graph Laplacian found: {0}'.format(w))

    return v[:,order[n]]
