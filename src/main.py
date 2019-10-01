#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../src")
import preprocessing
import cconfig
import clustering

# Plotting defaults

import matplotlib.pyplot as plt


# # Load data and preprocess

# In[2]:


# load original data in dataframes, sample, select some features and scale
df,df_Normal,df_Attack=preprocessing.data_load(1,None ,False)
X=preprocessing.data_scale(df[cconfig.SELECTED_FEATURES_FLOW])
X_Normal=preprocessing.data_scale(df_Normal[cconfig.SELECTED_FEATURES_FLOW])
X_Attack=preprocessing.data_scale(df_Attack[cconfig.SELECTED_FEATURES_FLOW])

print("Total Data Points: Training:",len(df_Normal)," Test:",len(df_Attack))

# # KMEANS

# In[3]:

print("Kmeans: Tunning number of clusters")
# find the best number of clusters
#df_silhouette = clustering.kmeans_get_number_clusters(X_Normal)

# select best number of clusters for kmeans
#num_clusters=df_silhouette.iloc[df_silhouette.score.idxmax() ]['Num Clusters']

# plot the result for reference
#df_silhouette.plot(x='Num Clusters', y='score')


# In[5]:

# fit kmeans model with normal day data
print("Kmeans: Fitting")
kmeans=clustering.kmeans_train(X_Normal,5)
print("Kmeans: Fitting...DONE")

# predictions with attack dataset
print("Kmeans: Predicting")
labels=clustering.kmeans_predict(X_Attack,kmeans)
print("Kmeans: Predicting...DONE")

# dimensionality reduction
print("Kmeans: Reducing dimensionality for plotting")
XR=preprocessing.get_pc(X_Attack,2)
print("Kmeans: Reducing dimensionality for plotting...DONE")

# print results
clustering.clustering_print_results(df_Attack,labels,cconfig.SELECTED_FEATURES_FLOW,XR,True,True,'kmeans')



# # OPTIC

# In[ ]:


# define hyper params for optics
eps=0.5
min_samples=26

# predict using optics
print("Optic: Fitting and predicting")
labels=clustering.optics_fit_predict(X,min_samples,'dbscan', eps)

# do dimensionality reduction to plot
XR=preprocessing.get_pc(X,2)
print("Optic: Fitting and predicting...DONE")

# print and plot
clustering.clustering_print_results(df,labels,cconfig.SELECTED_FEATURES_FLOW,XR,True,True,'optic')


# # IFOREST

# In[ ]:


# model iforest
print("IFOREST: Fitting and predicting")
iforest=clustering.iforest_train(X_Normal)
labels=clustering.iforest_predict(X_Attack,iforest)

# dimensionality reduction
XR=preprocessing.get_pc(X_Attack,2)
print("IFOREST: Fitting and predicting...DONE")
# print results
clustering.clustering_print_results(df_Attack,labels,cconfig.SELECTED_FEATURES_FLOW,XR,True,True,'iforest')


# In[ ]:




