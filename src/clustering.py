import utils
import pandas as pd

import sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

def kmeans_train(X,n_clusters=1):
    """Perform Kmeans clustering
    Parameters
    ----------
    X : array, shape (n_samples, n_features), or (n_samples, n_samples) 

    Returns
    -------
    self : instance of kmeans model
        The instance.
    """
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    return kmeans

def kmeans_predict(X,model):
    """Predict the closest cluster each sample in X belongs to.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        New data to predict.
    Returns
    -------
    labels : array, shape [n_samples,]
        Index of the cluster each sample belongs to.
    """
    labels=model.predict(X)
    return labels

def clustering_print_results(original_df,labels, features, X=None,print_out=True, plot_out=False):
    """Prints and plots clustering results.
    
    Parameters
    ----------
    X           : Matrix, shape = [n_samples, n_features]
    labels      : Predicted labels
    features    : Selected features
    original_df : Original dataframe with data
    print_out   : Print
    plot_out    : Export figure

    Returns
    -------
    labels : array, shape [n_samples,]
        Index of the cluster each sample belongs to.
    """
    # update original dataframe
    original_df['cluster'] = labels

    # group clusters
    clusters = original_df.groupby('cluster')

    if (plot_out):
        # data is expected to be reduced using dimensionality reduction
        original_df['x'] = X[:, 0] 
        original_df['y'] = X[:, 1] 

        
        # Define default colors
        colors = {-1:'black', 0:'green', 1:'blue', 2:'red', 3:'orange', 4:'purple', 5:'brown', 6:'pink', 7:'lightblue', 8:'grey', 9:'yellow'}
        fig, ax = plt.subplots()
        for key, cluster in clusters:
            cluster.plot(ax=ax, kind='scatter', x='x', y='y', alpha=0.5, s=250,label='Cluster: {:d}'.format(key), color=colors[key])
            fig.savefig("../outputs/clustering/"+str(utils.get_timestamp())+".png")

    if (print_out):
        for key, cluster in clusters:
            print('\nCluster {:d}: {:d} data points'.format(key, len(cluster)))
            print(cluster.head(3))

    print("done.")


def dbscan_fit_predict(eps,min_samples,X):
    """Perform DBSCAN clustering from features or distance matrix.
    Parameters
    ----------
    X : matrix of shape (n_samples, n_features)
    
    Returns
    ----------
    labels: Prediction
    """
    db=DBSCAN(eps=eps, min_samples=min_samples,algorithm='ball_tree', metric='euclidean').fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Total points: %d' % len(X))

    return labels

#optics_fit_predict(min_samples=50, xi=0.05, min_cluster_size=0.05, eps=2, X):

def optics_fit_predict(min_samples, xi, min_cluster_size, eps, X):
    """Perform OPTICS clustering
    Extracts an ordered list of points and reachability distances, and
    performs initial clustering using ``max_eps`` distance specified at
    OPTICS object instantiation.
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features), or (n_samples, n_samples)  

    Returns
    -------
    labels: Prediction/labels  
    """
    opt= OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    opt.fit(X)
    labels=cluster_optics_dbscan(reachability=opt.reachability_,
                                   core_distances=opt.core_distances_,
                                   ordering=opt.ordering_, eps=eps)
    return labels