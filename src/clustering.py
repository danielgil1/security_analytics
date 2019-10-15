import utils
import pandas as pd


import sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from scipy.stats import zscore

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from random import randint
from random import sample
import matplotlib.colors as pltc
from mpl_toolkits.mplot3d import Axes3D

def kmeans_fit_predict(X_train,X_test,n_clusters):
    """Perform Kmeans clustering
    Parameters
    ----------
    X : array, shape (n_samples, n_features), or (n_samples, n_samples) 

    Returns
    -------
    self : instance of kmeans model
        The instance.
    """
    print("Training kmeans with clusters: ",n_clusters)
    kmeans = KMeans(n_clusters=n_clusters).fit(X_train)
    labels=kmeans.predict(X_test)
    return kmeans,labels

def kmeans_train(X,n_clusters):
    """Perform Kmeans clustering
    Parameters
    ----------
    X : array, shape (n_samples, n_features), or (n_samples, n_samples) 

    Returns
    -------
    self : instance of kmeans model
        The instance.
    """
    print("Training kmeans with clusters: ",n_clusters)
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

def kmeans_get_number_clusters(X,max_clusters=7):
    """Get the score for a range number of clusters for kmeans using silhouette technique.
    
    Parameters
    ----------
    X : Matrix, shape = [n_samples, n_features]
        
    Returns
    -------
    df_silhouette : dataframe, shape [n_cluster,2]
        Index of the cluster each sample belongs to.
    """
    scores = []
    clusters = range(2,max_clusters+1)
    for K in clusters:
        
        clusterer = KMeans(n_clusters=K)
        cluster_labels = clusterer.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        scores.append(score)
        
    # Plot it out
    df_silhouette = pd.DataFrame({'Num Clusters':clusters, 'score':scores})
    return df_silhouette
    
def kmeans_anomalies_proximity(X,model,top=100):

    # identify the 5 closest points
    distances = model.transform(X)

    # argsort returns an array of indexes which will sort the array
    # in ascending order. Reverse it with [::-1]
    #sorted_idx = np.argsort(distances.ravel())[::-1][:5]
    sorted_idx = np.argsort(np.amax(distances,axis=1))[::-1][:top]
    return sorted_idx


def kmeans_anomalies_extreme_values(df,X,model,labels):
    # extreme value analysis
    distances = model.transform(X)

    df_anomalies_kmeans_zscore=pd.DataFrame(distances,columns=list(set(labels)))
    df_anomalies_kmeans_zscore=df_anomalies_kmeans_zscore.apply(zscore)
    df_kmeans_z=df.reset_index(drop=True)
    df_kmeans_z['zscore_0']=df_anomalies_kmeans_zscore[0]
    df_kmeans_z['zscore_1']=df_anomalies_kmeans_zscore[1]
    df_anomalies_kmeans_z=df_kmeans_z[((df_kmeans_z.zscore_0>3) | (df_kmeans_z.zscore_0<-3)) & ((df_kmeans_z.zscore_1>3) | (df_kmeans_z.zscore_1<-3)) ]
    
    return df_anomalies_kmeans_z

def clustering_print_results(original_df,labels, features, X=None,print_out=True, plot_out=False,label="clustering"):
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
    print("\nExporting "+label.upper()+"...")
    # update original dataframe
    original_df['cluster'] = labels

    # group clusters
    clusters = original_df.groupby('cluster')
    clusters_log=pd.DataFrame(columns=original_df.columns)
    print("Number of clusters:",len(set(labels)))
    if (plot_out):
        # data is expected to be reduced using dimensionality reduction
        original_df['x'] = X[:, 0] 
        original_df['y'] = X[:, 1] 
        
        # Define colors
        all_colors = [k for k,v in pltc.cnames.items()]
        colors = sample(all_colors, len(clusters))
        fig, ax = plt.subplots()
        indexColor=0
        for key,cluster in clusters:
            cluster.plot(ax=ax, kind='scatter', x='x', y='y', alpha=0.5, s=10,label='Cluster: {:d}'.format(key), color=colors[indexColor])
            indexColor+=1
        fig.savefig("../outputs/clustering/"+label+"_"+str(utils.get_timestamp())+".png")

    if (print_out):
        for key, cluster in clusters:
            print('\nCluster {:d}: {:d} data points'.format(key, len(cluster)))
            clusters_log=pd.concat([clusters_log,cluster.head(5)],sort=False)
        clusters_log.to_csv("../outputs/clustering/"+label+"_"+str(utils.get_timestamp())+".csv")
    
    print("\n"+"DONE.")
    print("-------------------------------------------------------")


def clustering_print_results_3d(original_df,labels, features, X=None,print_out=True, plot_out=False,label="clustering"):
    """Prints and plots 3d clustering results.
    
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

    print("\nExporting "+label.upper()+"...")
    # update original dataframe
    original_df['cluster'] = labels

    # group clusters
    clusters = original_df.groupby('cluster')
    clusters_log=pd.DataFrame(columns=original_df.columns)
    print("Number of clusters:",len(set(labels)))
    if (plot_out):
        # data is expected to be reduced using dimensionality reduction
        original_df['x'] = X[:, 0] 
        original_df['y'] = X[:, 1] 
        original_df['z'] = X[:, 2] 
        
        # Define colors
        all_colors = [k for k,v in pltc.cnames.items()]
        colors = sample(all_colors, len(clusters))
        print("color",len(colors))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = [x+2 for x in labels]
        ax.scatter3D(X[:,0], X[:,1], X[:,2], c=colors, cmap='Greens')
        
        
        #for key,cluster in clusters:
        #    ax.scatter(X[:,0], X[:,1], X[:,2], alpha=0.5, s=10,label='Cluster: {:d}'.format(key), color=colors[key])
            
        ax.view_init(15, 250)
        fig.savefig("../outputs/clustering/"+label+"_"+str(utils.get_timestamp())+".png")

    if (print_out):
        for key, cluster in clusters:
            print('\nCluster {:d}: {:d} data points'.format(key, len(cluster)))
            clusters_log=pd.concat([clusters_log,cluster.head(5)],sort=False)
        clusters_log.to_csv("../outputs/clustering/"+label+"_"+str(utils.get_timestamp())+".csv")
    
    print("\n"+"DONE.")
    print("-------------------------------------------------------")

def dbscan_fit_predict(eps,min_samples,X):
    """Perform DBSCAN clustering from features or distance matrix.
    Parameters
    ----------
    X           : matrix of shape (n_samples, n_features)
    eps         : The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself
    
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

def optics_fit_predict(X,min_samples=50, cluster_method='dbscan', eps=2):
    """Perform OPTICS clustering
    Extracts an ordered list of points and reachability distances, and
    performs initial clustering using ``max_eps`` distance specified at
    OPTICS object instantiation.
    
    Parameters
    ----------
    X               : array, shape (n_samples, n_features), or (n_samples, n_samples)  
    min_samples     : The number of samples in a neighborhood for a point to be considered as a core point.
    cluster_method  : 'dbscan' by default. Other available: 'xi'
    eps             : The maximum distance between two samples for one to be considered as in the neighborhood of the other.

    Returns
    -------
    labels: Prediction/labels  
    """
    opt= OPTICS(min_samples=min_samples, cluster_method=str(cluster_method))
    opt.fit(X)
    labels=cluster_optics_dbscan(reachability=opt.reachability_,
                                   core_distances=opt.core_distances_,
                                   ordering=opt.ordering_, eps=eps)

    return labels

def optics_anomalies(original_df,labels):

    original_df['cluster'] = labels
    # get outliers as cluster -1

    return original_df[original_df.cluster==-1]

def iforest_train(X,contam=0.25):
    
    iforest = IsolationForest(behaviour='new', contamination=contam)
    iforest.fit(X)
    return iforest

def iforest_predict(X,model):
    labels=model.predict(X)
    return labels

def iforest_anomalies(original_df,labels):

    original_df['cluster'] = labels
    # get outliers as cluster -1
    
    return original_df[original_df.cluster==-1]

def lof_fit_predict(X,outliers_fraction=0.10,n_neighbors=35):
    lof_model=LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
    labels=lof_model.fit_predict(X)
    return labels

def lof_anomalies(original_df,labels):

    original_df['cluster'] = labels
    # get outliers as cluster -1
    
    return original_df[original_df.cluster==-1]

def ocsvm_fit_predict(X_Train,X_Test,outliers_fraction=0.10):
    model_ocsvm=svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",gamma=0.1)
    labels=model_ocsvm.fit(X_Train).predict(X_Test)
    return labels

def ocsvm_anomalies(original_df,labels):

    original_df['cluster'] = labels
    # get outliers as cluster -1
    
    return original_df[original_df.cluster==-1]
