import os
import json

import numpy as np
from tslearn.clustering import TimeSeriesKMeans

def save_model(model, cloud_ids, model_id, model_path="./cluster_models", overwrite=True):
    """saves trained model to disk
    Args:
        model (_type_): trained model
        cloud_ids (np.ndarray): array of cloud ids the model was trained on
        model_id (str): identifier for model
        model_path (str): directory where models are saved
        overwrite (bool, optional): overwrite flag. Defaults to True.
    """
    # create directory
    path = "{}/{}".format(model_path, model_id)
    try:
        os.makedirs(path, mode=0o755)
        print("created:", path)
    except FileExistsError:
        print("directory `{}` already exists".format(path))
        if overwrite:
            print("overwrite model and params")
            os.remove(path + '/trained_model.hdf5')
            pass
        else:
            print("abort")
            return
        
    # save model
    model.to_hdf5(path + '/trained_model.hdf5')
    print("saved model to file")
    
    # save model params
    with open(path + "/model_params.json", "w") as file:
        json.dump(model.get_params(), file)
    
    # save cloud_ids
    np.save(path + "/cloud_ids.npy",cloud_ids)
    print("saved cloud_ids to disk")


def load_model(model_id, model_path="./cluster_models"):
    """load model from disk

    Args:
        model_id (str): model identifier
        model_path (str): directory where models are saved

    Returns:
        TimeSeriesKMeans, np.ndarray: trained clustering model, cloud ids the model was trained on
    """
    path = "{}/{}".format(model_path, model_id)
    
    # load model params
    with open(path + "/model_params.json", "r") as file:
        model_params = json.load(file)
        
    # init model 
    model = TimeSeriesKMeans(**model_params).from_hdf5(path + "/trained_model.hdf5")
    
    try:
        # load training data
        cloud_ids = np.load(path + "/cloud_ids.npy", allow_pickle=True)
    except FileNotFoundError:
        # load training data
        cloud_ids = np.load(path + "/traj_ids.npy", allow_pickle=True)
        
    print("sample cloud_id:", cloud_ids[0])
        
    
    return model, cloud_ids


def run_cluster(n_clusters, X, cloud_ids, clustering_kws, model_id, model_path="./cluster_models"):
    """

    Args:
        n_clusters (int): number of clusters (k)
        X (_type_): _description_
        cloud_ids (np.ndarray): _description_
        clustering_kws (dict): _description_
        model_id (str): model identifier
        model_path (str): directory where model is stored
    """
    print(n_clusters, X.shape)
    km_dba = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, **clustering_kws)
    cluster_pred = km_dba.fit_predict(X) 
    save_model(km_dba, cloud_ids, model_id,model_path=model_path)
    print("saved model")

    return km_dba

