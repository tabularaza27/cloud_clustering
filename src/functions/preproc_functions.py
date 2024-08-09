import numpy as np
import pandas as pd

from helper_functions import round_to_multiple
from helper_functions import percentile
from helper_functions import sel_cloud_center

JOIN_VARS = ["trajectory_id","dz_top_v2","lev","iwc","icnc_5um","icnc_100um","iwc_log","icnc_5um_log","icnc_100um_log","reffcli","season","region","nightday_flag","T_rounded","OMEGA_rounded","DU_sup_log","DU_sub_log"] # cloud_thickness_v2

def get_pred_dfs(start_points_df, traj_df_, traj_array, cluster_pred,var="dust"):
    """generating 2 dataframes containing info about trajectory data, cloud data, and cluster predictions

    traj_df: contains data along whole trajectory & corresponding cluster prediction (i.e. one row represents one timestep)
    cluster_pred_df: contains info at trajectory startpoints (i.e. one row represents one cloud)

    Args:
        start_points_df (_type_): _description_
        traj_df_ (_type_): _description_
        traj_array (_type_): _description_
        cluster_pred (_type_): _description_
        var (str, optional): _description_. Defaults to "dust".

    Returns:
        _type_: _description_
    """
    col_ids = np.where(traj_df_.columns.isin(['trajectory_id_top','cloud_id','trajectory_id_base','trajectory_id_center']))[0]
    cluster_pred_df = pd.DataFrame(np.concatenate((traj_array[:,0,col_ids],np.expand_dims(cluster_pred,1)),1),columns=["cloud_top_traj_id","cloud_id","cloud_base_traj_id","cloud_center_traj_id","cluster_idx"])
    
    # join info about top and base level
    _ = pd.merge(cluster_pred_df, start_points_df[start_points_df.trajectory_id.isin(cluster_pred_df.cloud_top_traj_id)][JOIN_VARS], left_on="cloud_top_traj_id",right_on="trajectory_id",suffixes=("","_top"))
    cluster_pred_df = pd.merge(_, start_points_df[start_points_df.trajectory_id.isin(cluster_pred_df.cloud_base_traj_id)][JOIN_VARS], left_on="cloud_base_traj_id",right_on="trajectory_id",suffixes=("_top","_base"))

    if var=="dust": 
        c_map = {c_idx: c_name for c_idx, c_name in zip(np.argsort(np.mean(model.cluster_centers_,1).squeeze()),["1_low","2_medium","3_high"])} # renaming clusters to intrepretable names corresponding to low, medium, high dust concentrations
        cluster_pred_df = cluster_pred_df.replace({"cluster_idx": c_map})
    elif var=="omega":
        c_map = {c_idx: c_name for c_idx, c_name in zip(np.flip(np.argsort(np.mean(model.cluster_centers_,1).squeeze())),["1_low","2_medium","3_high"])} # renaming clusters to intrepretable names corresponding to low, medium, high dust concentrations
        cluster_pred_df = cluster_pred_df.replace({"cluster_idx": c_map})
        
    traj_df = pd.merge(traj_df_, cluster_pred_df.drop(["trajectory_id_top","trajectory_id_base","T_rounded_base","T_rounded_top","OMEGA_rounded_base","OMEGA_rounded_top","DU_sup_log_base","DU_sup_log_top","nightday_flag_top","nightday_flag_base","region_top"],1), on="cloud_id")
    traj_df = traj_df.rename(columns={"region_base":"region"})
    
    return cluster_pred_df, traj_df


def get_whole_cloud_df(start_points_df, cluster_pred_df):
    
    whole_cloud_df = start_points_df[start_points_df.cloud_id.isin(cluster_pred_df.cloud_id.unique())][['lonr', 'latr', 'cloud_id', 'timestep'] + JOIN_VARS]
    cluster_pred_df_join_cols = ["cloud_id","cluster_idx"]
    if "omega_cluster_idx" in cluster_pred_df.columns: cluster_pred_df_join_cols.append("omega_cluster_idx")
    whole_cloud_df = pd.merge(whole_cloud_df, cluster_pred_df[cluster_pred_df_join_cols], on="cloud_id")
    print(whole_cloud_df.cluster_idx.value_counts())
    whole_cloud_df["dz_top_v2_rounded"] = round_to_multiple(whole_cloud_df.dz_top_v2,300)
    # whole_cloud_df["dz_top_rel"] = round_to_multiple(whole_cloud_df.dz_top_v2 / whole_cloud_df.cloud_thickness_v2, 0.333)

    cloud_top = start_points_df.drop_duplicates("cloud_id",keep="first").query("dz_top_v2<400")
    whole_cloud_df = pd.merge(whole_cloud_df, cloud_top[["cloud_id", "iwc", "icnc_5um", "icnc_100um", "reffcli", "T"]], how="left", on="cloud_id", suffixes=('','_top'))
    
    # calculate relative deviance of cloud layer cirrus var to cloud top
    #for cirrus_var in ["iwc", "icnc_5um", "icnc_100um", "reffcli"]:
    #    whole_cloud_df[f"{cirrus_var}_rel_to_top"] = (whole_cloud_df[cirrus_var] / whole_cloud_df[f"{cirrus_var}_top"]).round(1)
    #        
    return whole_cloud_df


def prep_cluster_data(df, clustering_features, cloud_ids):
    """get input data for clustering, i.e. trajectories starting at cloud top, center, base 

    Args:
        df (pd.DataFrame): dataframe containing cloud and trajectory data
        clustering_features (list): features used for clustering
        cloud_ids (np.ndarray): cloud ids of clouds that will be clustered

    Returns:
        pd.DataFrame (traj_df) : dataframe where each row represents a whole cloud with cloud top, center, base
        np.ndarray (traj_array): same as above as numpy array
        np.ndarray (X): same as above, but just containing columns of features used for clustering
    """
    start_points_df = df.query("timestep==0")

    cloud_df = start_points_df[start_points_df.cloud_id.isin(cloud_ids)].query("cloud_cover>=0.1")

    # select cloud tops 
    # quick fix for error described above is filtering for dz_top_v2
    cloud_top = cloud_df.drop_duplicates("cloud_id",keep="first").query("dz_top_v2<400")

    cloud_base = cloud_df[cloud_df.cloud_id.isin(cloud_top.cloud_id)].drop_duplicates("cloud_id",keep="last")

    # select cloud center for omega clustering
    cloud_center = cloud_df[cloud_df.cloud_id.isin(cloud_top.cloud_id)].groupby('cloud_id', as_index=False).apply(sel_cloud_center)

    cloud_top_traj_ids = cloud_top.trajectory_id.values
    cloud_base_traj_ids = cloud_base.trajectory_id.values
    cloud_center_traj_ids = cloud_center.trajectory_id.values

    # get trajectory data as numpy array
    features = ["T","OMEGA","IWC","LWC","RH_ice","DU_sup","DU_sub","DU_sup_log"]#,"OMEGA","RH_ice"] # "T","OMEGA","DU_log", "SO4_traj", "IC_CIR_class"
    helper_variables = ["lonr","latr","trajectory_id","lon","lat","nightday_flag"] # todo remove lon,lat
    join_variables = ["timestep", "cloud_id"]

    traj_cloud_top = df[df.trajectory_id.isin(cloud_top_traj_ids)][features + helper_variables + join_variables]
    traj_cloud_base = df[df.trajectory_id.isin(cloud_base_traj_ids)][features + helper_variables + join_variables]
    traj_cloud_center = df[df.trajectory_id.isin(cloud_center_traj_ids)][features + helper_variables + join_variables]

    # df containing cloud top and cloud base and cloud center values
    traj_df = pd.merge(traj_cloud_top, traj_cloud_base, on=["cloud_id","timestep"],suffixes=("_top","_base"))
    traj_df = pd.merge(traj_df, traj_cloud_center, on=["cloud_id","timestep"],suffixes=("_top","_center"))
    traj_df = traj_df.rename(columns={c:c+"_center" for c in features + helper_variables})

    # numpy array for training
    traj_array = traj_df.values.reshape(cloud_top_traj_ids.shape[0], 24, len(traj_df.columns)) # n_trajectories, timesteps, n_columns
     
    col_ids = np.where(traj_df.columns.isin(clustering_features))[0]

    X = traj_array[:,:,col_ids]
     
    return traj_df, traj_array, X

def prep_preprocessed_cluster_data(traj_df, clustering_features):
    """same as prep_cluster_data but with precomputed dataframe where each row represents a t timestep at cloud top, center, and base

    Args:
        traj_df (pd.DataFrame): generated by `prep_cluster_data()`
        clustering_features (list): features used for clustering

     Returns:
        np.ndarray (traj_array): same as above as numpy array
        np.ndarray (X): same as above, but just containing columns of features used for clustering
    """

    # numpy array for training
    traj_array = traj_df.values.reshape(traj_df.cloud_id.unique().shape[0], 24, len(traj_df.columns)) # n_trajectories, timesteps, n_columns     
    col_ids = np.where(traj_df.columns.isin(clustering_features))[0]
    X = traj_array[:,:,col_ids]

    return traj_array, X

   