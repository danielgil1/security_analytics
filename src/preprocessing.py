import cconfig
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pickle


def data_load(sample_frac=1,features=None,scale=False,dataset_type=cconfig.DATASET_TYPE_BIFLOW):
    """Loads data into dataframe and performs sampling if needed
    
    Parameters
    ----------
    sample_frac : % of data to be sampled. Range [0,1].   

    Returns
    -------
    df_all      : Dataframe with train and test set
    df_train    : Dataframe with training set
    df_test     : Dataframe with test set 
    """
    if (dataset_type==cconfig.DATASET_TYPE_FLOW):
        training_dataset=cconfig.TRAIN_DATA_FLOW
        testing_dataset=cconfig.TEST_DATA_FLOW
    elif (dataset_type==cconfig.DATASET_TYPE_PACKET):
        training_dataset=cconfig.TRAIN_DATA_PACKET
        testing_dataset=cconfig.TEST_DATA_PACKET
    else:
        training_dataset=cconfig.TRAIN_DATA_BIFLOW
        testing_dataset=cconfig.TEST_DATA_BIFLOW

    if features==None:
        if (dataset_type!=cconfig.DATASET_TYPE_BIFLOW):
            df_train=pd.read_csv(training_dataset).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE).fillna(0)
            df_test=pd.read_csv(testing_dataset).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE).fillna(0)
        else:
            df_train=(pickle.load(open(training_dataset, "rb" ))).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE).fillna(0) 
            df_test=(pickle.load(open(testing_dataset, "rb" ))).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE).fillna(0)
    else:
        if (dataset_type!=cconfig.DATASET_TYPE_BIFLOW):
            df_train=pd.read_csv(training_dataset).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE)[features].fillna(0)
            df_test=pd.read_csv(testing_dataset).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE)[features].fillna(0)
        else:
            
            df_train=pickle.load(open(training_dataset, "rb" )).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE)[features].fillna(0) 
            df_test=pickle.load(open(testing_dataset, "rb" )).sample(frac=sample_frac, replace=False, random_state=cconfig.RANDOM_STATE)[features].fillna(0)

    
    df_all=pd.concat([df_train,df_test])
    
    if scale:
        return data_scale(df_all),data_scale(df_train),data_scale(df_test)
    else:
        return df_all,df_train,df_test



def data_scale(data):
    """Transform dataframe to a matrix of scaled numbers
    
    Parameters
    ----------
     

    Returns
    -------
    
    
    """
    return StandardScaler().fit_transform(data)

def get_pc(X,num_components=2,tsne=False):
    """Performs dimensionality reduction to a matrix
    
    Parameters
    ----------
        num_components  : New dimensions
        tsne            : True if reduction with TSNE, PCA otherwise

    Returns
    -------
    
    
    """
    if tsne:
        reduced_data = TSNE(n_components=num_components).fit_transform(X)
    else:
        pca = PCA(n_components=num_components,random_state=cconfig.RANDOM_STATE).fit(X)
        print("Variability explained by the PC:",sum(pca.explained_variance_ratio_) )
        reduced_data = pca.transform(X)
    
    return reduced_data

