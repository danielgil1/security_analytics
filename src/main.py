import preprocessing
import cconfig
import clustering
import utils

# Plotting defaults
import matplotlib.pyplot as plt


# # Load data and preprocess

# select type of features and default values
selected_features=cconfig.SELECTED_FEATURES_BIFLOW
dataset_type=cconfig.DATASET_TYPE_BIFLOW
max_num_clusters=cconfig.DEFAULT_NUM_CLUSTERS
sort_anomalies=cconfig.BIFLOW_ANOMALIES_SORT


# # Bidirectional flow

# load original data in dataframes, sample, select some features and scale
df,df_Normal,df_Attack=preprocessing.data_load(1,None,False,dataset_type)
X=preprocessing.data_scale(df[selected_features])
X_Normal=preprocessing.data_scale(df_Normal[selected_features])
X_Attack=preprocessing.data_scale(df_Attack[selected_features])


# # KMEANS

# find the best number of clusters
#df_silhouette = clustering.kmeans_get_number_clusters(X_Normal)

# select best number of clusters for kmeans
max_num_clusters=4 #df_silhouette.iloc[df_silhouette.score.idxmax() ]['Num Clusters']


# fit kmeans model with normal day data
kmeans=clustering.kmeans_train(X_Normal,int(max_num_clusters))

# predictions with attack dataset
labels=clustering.kmeans_predict(X_Attack,kmeans)

# dimensionality reduction
XR=preprocessing.get_pc(X_Attack,2)

# print results
clustering.clustering_print_results(df_Attack,labels,selected_features,XR,True,True,dataset_type+'_kmeans')


# In[17]:


# print anomalies
index_anomalies=clustering.kmeans_anomalies(X_Attack,kmeans)
df_anomalies_kmeans=df_Attack.iloc[index_anomalies,:]
df_anomalies_kmeans.sort_values(by=sort_anomalies,ascending=False)
utils.save(df_anomalies_kmeans,"df_anomalies_kmeans")

# # DBSCAN

# define hyper parameters for dbscan
eps=0.5
min_samples=26

# fit and predict
#dblabels=clustering.dbscan_fit_predict(eps,min_samples,X)

# do dimensionality reduction to plot
#XR=preprocessing.get_pc(X,2)

# print and plot
#clustering.clustering_print_results(df,dblabels,selected_features,XR,True,True,dataset_type+'_dbscan')


# # OPTIC


# define hyper params for optics
eps=1.5
min_samples=20

# predict using optics
labels=clustering.optics_fit_predict(X,min_samples,'dbscan', eps)

# do dimensionality reduction to plot
XR=preprocessing.get_pc(X,2)

# print and plot
clustering.clustering_print_results(df,labels,selected_features,XR,True,True,dataset_type+'_optic')


df_anomalies_optic=clustering.optics_anomalies(df,labels)
df_anomalies_optic.sort_values(by=sort_anomalies,ascending=False)
utils.save(df_anomalies_optic,"df_anomalies_optic")

# # IFOREST

# model iforest
iforest=clustering.iforest_train(X_Normal)
labels=clustering.iforest_predict(X_Attack,iforest)

# dimensionality reduction
XR=preprocessing.get_pc(X_Attack,2)

# print results
clustering.clustering_print_results(df_Attack,labels,selected_features,XR,True,True,dataset_type+'_iforest')


# get anomalies
df_anomalies_iforest=clustering.iforest_anomalies(df_Attack,labels)
df_anomalies_iforest.sort_values(by=sort_anomalies,ascending=False)
utils.save(df_anomalies_iforest,"df_anomalies_iforest")


# # LOF

outliers_fraction=0.05
n_neighbors=30
labels=clustering.lof_fit_predict(X,outliers_fraction,n_neighbors)

# dimensionality reduction
XR=preprocessing.get_pc(X,2)

# print results
clustering.clustering_print_results(df,labels,selected_features,XR,True,True,dataset_type+'_lof')

# get anomalies
df_anomalies_lof=clustering.lof_anomalies(df,labels)
df_anomalies_lof.sort_values(by=sort_anomalies,ascending=False)
utils.save(df_anomalies_lof,"df_anomalies_lof")

# # OCSVM

# train and test the model
outliers_fraction=0.05
labels=clustering.ocsvm_fit_predict(X_Normal,X_Attack,outliers_fraction)

# dimensionality reduction
XR=preprocessing.get_pc(X_Attack,3)

# print results
clustering.clustering_print_results(df_Attack,labels,selected_features,XR,True,True,dataset_type+'_ocsvm')

# get anomalies
df_anomalies_ocsvm=clustering.ocsvm_anomalies(df_Attack,labels)
df_anomalies_ocsvm.sort_values(by=sort_anomalies,ascending=False)
utils.save(df_anomalies_ocsvm,"df_anomalies_ocsvm")

