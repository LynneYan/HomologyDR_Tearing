from __future__ import division
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.datasets.samples_generator import make_s_curve
from sklearn import preprocessing
from shutil import copyfile

_DATA_DIR = '../Data'


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def create_folders(_DATA_TYPE):
    make_dir(_DATA_DIR)
    for data_type in _DATA_TYPE:
        data_dir = os.path.join(_DATA_DIR, data_type)
        make_dir(data_dir)
        copyfile('../Realworld_dataset/'+data_type+".txt", os.path.join(data_dir, 'X.txt'))
    
def _get_input_data(data_dir):
    in_file_X = os.path.join(data_dir, 'X.txt')
    in_file_color  = os.path.join(data_dir, 'color.txt')
    X = np.loadtxt(in_file_X)
    color = np.loadtxt(in_file_color)
    return X, color

def _get_normalized_input_data(data_dir):
    in_file_X = os.path.join(data_dir, 'X_norm.txt')
    in_file_color  = os.path.join(data_dir, 'color.txt')
    X = np.loadtxt(in_file_X)
    color = np.loadtxt(in_file_color)
    return X, color

def _rotate(x, y, z, noise):
    u =x
    cos = math.cos(0)
    sin = math.sin(0)
    v = cos * y + sin * z
    w = -sin * y +cos * z
    return [20*u-np.random.uniform(-1,1)*noise*100, 20*v-np.random.uniform(-1,1)*noise*100, 20*w-np.random.uniform(-1,1)*noise*100]

def _save_output_data(Y, Y_dir, file_name):
    out_file_Y = os.path.join(Y_dir, file_name + '.txt')
    np.savetxt(out_file_Y, Y, fmt='%f')


def _generate_LR(n_samples, noise):
    
    points = [0, 0, 0];
    color = [5.757057];
    for i in range (1, n_samples+1):
        t = 2* math.pi * i / n_samples
        sin = math.sin(t)
        cos = math.cos(t)
        points= np.vstack([points, _rotate(cos, sin, 0, noise)])
        color=np.vstack([color, i/213.3333])
        points= np.vstack([points, _rotate(1+cos, 0, sin, noise)])
        color=np.vstack([color, 7.5+i/213.333333])
    return points, color


def _plot_original_data(X, color, data_dir):
    Plot_dir = os.path.join(data_dir, 'Plot')
    make_dir(Plot_dir)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    cs = ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.Spectral_r, s=8, c= X[:,0])
    Plot_dir = os.path.join(Plot_dir, "X.png")
    plt.axis("equal")
    #plt.axis('off')
    plt.show()
    fig.savefig(Plot_dir)


    
def _generate_LR_data(n_samples_, noise, outlier_,exp):
    out_file = os.path.join(_DATA_DIR, 'Linked_circles_'+str(noise)+'_'+str(outlier_))
    make_dir(out_file)
    out_file = os.path.join(out_file, "Exp_"+str(exp))
    make_dir(out_file)
    n_samples = int((1 - outlier_)*n_samples_)
    X, color = _generate_LR(n_samples,noise)
    X = X[1:]
    color = color[1:]
    X = preprocessing.scale(X)
    outlier = int(outlier_*n_samples_)
    if outlier != 0:
        X = np.r_[X, np.random.uniform(low=-2, high=2, size=(outlier, 3))]
        color = np.r_[color, np.ones((outlier,1))*9]
    X_norm = preprocessing.RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    X_norm = preprocessing.MinMaxScaler(feature_range=(-2.5, 2.5)).fit_transform(X_norm)
    out_file_X_norm = os.path.join(out_file, 'X_norm.txt')
    out_file_X = os.path.join(out_file, 'X.txt')
    out_file_color  = os.path.join(out_file, 'color.txt')
    np.savetxt(out_file_X, X, fmt='%f')
    np.savetxt(out_file_color, color, fmt='%f')
    np.savetxt(out_file_X_norm, X_norm, fmt='%f')
    _plot_original_data(X_norm, color, out_file)
   

def _generate_U_SH_data(n_samples_,noise, outlier_,exp):
    out_file = os.path.join(_DATA_DIR, 'Swiss_hole_'+str(noise)+'_'+str(outlier_))
    make_dir(out_file)
    out_file = os.path.join(out_file, "Exp_"+str(exp))
    make_dir(out_file)
    n_samples = int((1 - outlier_)*n_samples_)
    X, color = make_swiss_roll(n_samples, noise)
    X = preprocessing.scale(X)
    outlier = int(outlier_*n_samples_)
    if outlier != 0:
        X = np.r_[X, np.random.uniform(low=-3, high=3, size=(outlier, 3))]
        color = np.r_[color, np.ones((outlier,))*13]
    X_norm = preprocessing.RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    X_norm = list(X_norm)
    X_norm = [item for item in X_norm if (abs(item[0])>0.25 or abs(item[1])>0.25) or item[2]>-0.6]
    X_norm = np.array(X_norm)
    X_norm = preprocessing.MinMaxScaler(feature_range=(-2.5, 2.5)).fit_transform(X_norm)
    out_file_X_norm = os.path.join(out_file, 'X_norm.txt')
    out_file_X = os.path.join(out_file, 'X.txt')
    out_file_color  = os.path.join(out_file, 'color.txt')
    np.savetxt(out_file_X, X, fmt='%f')
    np.savetxt(out_file_color, color, fmt='%f')
    np.savetxt(out_file_X_norm, X_norm, fmt='%f')
    _plot_original_data(X_norm, color, out_file)

def _generate_UFN_data(n_samples_, noise, outlier_, exp):
    if outlier_>0:
        n_samples_ = int(n_samples_*0.9)
    out_file = os.path.join(_DATA_DIR, 'Fishing_net_'+str(noise)+'_'+str(outlier_))
    make_dir(out_file)
    out_file = os.path.join(out_file, "Exp_"+str(exp))
    make_dir(out_file)
    n_samples = int((1 - outlier_)*n_samples_)
    X,color = make_s_curve(n_samples, noise)
    outlier = int(outlier_*n_samples_)
    X_norm = X
    X_norm = list(X_norm)
    r = 0.2
    h = 0.5
    for yc in [0.4, 1, 1.6]:
        for xc in [-0.6, 0, 0.6]:
            for zc in [-1.7, 0, 1.7]:
                X_norm = [item for item in X_norm if (abs(item[0]-xc)>r or abs(item[1]-yc)>r) or abs(item[2]-zc)>h]
    r = 0.2
    w = 0.15
    h = 0.3
    for yc in [0.4, 1, 1.6]:
        for xc in [-1.05, 1.05]:
            for zc in [-1, 1]:
                X_norm = [item for item in X_norm if (abs(item[0]-xc)>w or abs(item[1]-yc)>r) or abs(item[2]-zc)>h]
    X_norm = np.array(X_norm)

    X_norm = preprocessing.MinMaxScaler(feature_range=(-2.5, 2.5)).fit_transform(X_norm)
    if outlier != 0:
        X_norm = np.r_[X_norm, np.random.uniform(low=-3, high=3, size=(outlier, 3))]
        color = np.r_[color, np.ones((outlier,))*13]
    X_norm = preprocessing.MinMaxScaler(feature_range=(-2.5, 2.5)).fit_transform(X_norm)
    out_file_X_norm = os.path.join(out_file, 'X_norm.txt')
    out_file_X = os.path.join(out_file, 'X.txt')
    out_file_color  = os.path.join(out_file, 'color.txt')
    np.savetxt(out_file_X, X, fmt='%f')
    np.savetxt(out_file_color, color, fmt='%f')
    np.savetxt(out_file_X_norm, X_norm, fmt='%f')
    _plot_original_data(X_norm, X_norm[:, 0], out_file)
