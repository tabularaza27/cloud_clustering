import sys
sys.path.append(".")

import numpy as np

import holoviews as hv
import bokeh.palettes as palettes

from helper_functions import percentile


### kwargs / dicts / meta data ###

cluster_labels = {"0_in-situ": "in situ",
                  "1_hybrid": "hybrid", 
                  "2_liquid-origin_thin": "liquid origin (thin)", 
                  "3_liquid-origin_thick": "liquid origin (thick)",
                  "1_low": "low ω regime",
                  "2_medium": "medium ω regime",
                  "3_high": "high ω regime"}

temp_cluster_map = {1: '3_liquid-origin_thick',
                    0: '1_hybrid',
                    2: '0_in-situ',
                    3: '2_liquid-origin_thin'}

plt_options = {'fontsize': {'xlabel': 15,
      'ylabel': 15,
      'clabel': 15,
      'ticks': 15,
      'legend': 15,
      'title': 25}}

temp_colors = list(palettes.Spectral4)
omega_colors = tuple(np.flip(np.array(list(palettes.Purples9)))[[3,6,8]])

units = dict(iwc="[mg m⁻³]",iwc_log="log10 [mg m⁻³]", icnc_5um="[cm⁻³]",icnc_5um_log="log10 [cm⁻³]", dust="[mg kg-1]",T="[K]")
var_names = dict(iwc="IWC", icnc_5um="$N_{{ice}}$", dust="Dust > 1 μm",T="Temperature")

def get_cluster_traj_plts(traj_df,colors=list(np.flip(palettes.Oranges3)),subplots=True,cluster_col="cluster_idx"):
    
    cluster_timestep_stats = traj_df.groupby([cluster_col,"timestep"],as_index=False).agg(cluster_idx_count=("cluster_idx",'count'), 
                                               T_top_mean=("T_top",np.mean),
                                               T_top_std=("T_top",np.std),
                                               T_base_mean=("T_base",np.mean),
                                               T_base_std=("T_base",np.std),
                                               OMEGA_center_mean=("OMEGA_center",np.mean),
                                               OMEGA_center_std=("OMEGA_center",np.std))

    cluster_timestep_stats["T_top_upper"] = cluster_timestep_stats.T_top_mean + cluster_timestep_stats.T_top_std
    cluster_timestep_stats["T_top_lower"] = cluster_timestep_stats.T_top_mean - cluster_timestep_stats.T_top_std

    cluster_timestep_stats["T_base_upper"] = cluster_timestep_stats.T_base_mean + cluster_timestep_stats.T_base_std
    cluster_timestep_stats["T_base_lower"] = cluster_timestep_stats.T_base_mean - cluster_timestep_stats.T_base_std

    cluster_timestep_stats["OMEGA_center_upper"] = cluster_timestep_stats.OMEGA_center_mean + cluster_timestep_stats.OMEGA_center_std
    cluster_timestep_stats["OMEGA_center_lower"] = cluster_timestep_stats.OMEGA_center_mean - cluster_timestep_stats.OMEGA_center_std
    
    cluster_timestep_median = traj_df.groupby([cluster_col,"timestep"]).median().reset_index()
    
    ### temp plot ###
    variable = "T"

    c_plts = []
    for i,cluster_idx in enumerate(cluster_timestep_median[cluster_col].unique()):
        print(i, cluster_idx)
        c_df = cluster_timestep_stats.query(f"{cluster_col}=='{cluster_idx}'")
        c_plt = (c_df.hvplot.line(title=cluster_labels[cluster_idx],x="timestep",y=f"{variable}_top_mean", line_dash="dashed",line_width=10,height=400,width=400,color=colors[i],legend=False,**plt_options,ylabel="Temperature [K]", xlabel="Timestep [h]",ylim=(180,280)) * 
                 c_df.hvplot.errorbars(x="timestep",y=f"{variable}_top_mean",yerr1=f"{variable}_top_std",color=colors[i],alpha=0.5,ylim=(180,280)) *
                 c_df.hvplot.line(x="timestep",y=f"{variable}_base_mean", line_width=10,color=colors[i],legend=False,ylim=(180,280)) * 
                 c_df.hvplot.errorbars(x="timestep",y=f"{variable}_base_mean",yerr1=f"{variable}_base_std",color=colors[i],alpha=0.5,ylim=(180,280)) *
                 c_df.hvplot.area(x="timestep",y=f"{variable}_top_mean",y2=f"{variable}_base_mean",alpha=0.2,stacked=False,color=colors[i],legend=False,ylim=(180,280)) *
                 hv.HLine(235.15).opts(color="black",line_width=1)).opts(invert_yaxis=True)
        c_plts.append(c_plt)
    temp_plt = hv.Layout(c_plts)
    
    
    ### omega plot ###
    omega_c_plts = []
    for i,cluster_idx in enumerate(cluster_timestep_median[cluster_col].unique()):
        # print(i)
        c_df = cluster_timestep_stats.query(f"{cluster_col}=='{cluster_idx}'")
        omega_c_plt = (c_df.hvplot.line(x="timestep",y=f"OMEGA_center_mean", title=cluster_labels[cluster_idx], line_width=5,height=200,width=400,color=colors[i],legend=False,subplots=subplots,**plt_options,ylabel="ω [Pa s⁻¹]", xlabel="Timestep [h]").opts(invert_yaxis=True) * 
                       c_df.hvplot.errorbars(x="timestep",y=f"OMEGA_center_mean",yerr1="OMEGA_center_std",color=colors[i],alpha=0.5))
        omega_c_plts.append(omega_c_plt)
    omega_plt = hv.Layout(omega_c_plts)
    
    ### dust plot ###
    # omega_plt = (cluster_timestep_median.hvplot.line(x="timestep",y=f"OMEGA_center",by="cluster_idx", width=500, height=400,subplots=False,legend="bottom",color=cmap,line_width=4,**plt_options))
    # temp_plt = (traj_df.groupby(["cluster_idx","timestep"]).median().reset_index().hvplot.area(x="timestep",y="T_top",y2="T_base",alpha=0.5,by="cluster_idx",stacked=False, width=500, height=400,subplots=False,legend="bottom",color=cmap,**plt_options))# * hv.HLine(235.15).opts(color="black"))#.cols(6)
    variable = "DU_sup"
    dust_plt = (cluster_timestep_median.hvplot.area(x="timestep",y=f"{variable}_top",y2=f"{variable}_base",alpha=0.5,by=cluster_col,stacked=False, width=400, height=400,subplots=False,legend="bottom",color=colors,**plt_options,logy=True))#.cols(6)
    
    
    # display(cluster_pred_df.groupby("cluster_idx").median()[["T_top","OMEGA_top","DU_sup_log_top", "iwc_top", "icnc_5um_top", "icnc_100um_top", "reffcli_top","T_base","OMEGA_base","cloud_thickness_v2_base", "lev_base", "iwc_base", "icnc_5um_base", "icnc_100um_base", "reffcli_base"]])
    # display(cluster_pred_df.groupby(["cluster_idx","season_top"]).count()[["T_top","OMEGA_top"]])
    # display(cluster_pred_df.cluster_idx.value_counts())
    return temp_plt, omega_plt, dust_plt

def get_whole_cloud_parameter_space_plts(whole_cloud_df, colors=list(np.flip(palettes.Oranges3)),quantile_lines=True,min_count=1000,logy=False,cluster_col="cluster_idx",max_dz_top_v2=10000):
    
    whole_cloud_parameter_space = whole_cloud_df.query(f"dz_top_v2 <= {max_dz_top_v2}").groupby([f"T_rounded",cluster_col]).agg(count=("cloud_id",'count'), 
                                               iwc_lower=(f"iwc",percentile(25)),
                                               iwc_median=(f"iwc",np.median),
                                               iwc_upper=(f"iwc",percentile(75)),
                                               icnc_5um_lower=(f"icnc_5um",percentile(25)),
                                               icnc_5um_median=(f"icnc_5um",np.median),
                                               icnc_5um_upper=(f"icnc_5um",percentile(75)),
                                               # icnc_100um_lower=(f"icnc_100um",percentile(25)),
                                               # icnc_100um_median=(f"icnc_100um",np.median),
                                               # icnc_100um_upper=(f"icnc_100um",percentile(75)),
                                               reffcli_median = ("reffcli",np.median)).reset_index().query(f"count>{min_count}")
    whole_cloud_parameter_space=whole_cloud_parameter_space.sort_values([cluster_col, "T_rounded"])

    all_data_whole_cloud_parameter_space = whole_cloud_df.query(f"dz_top_v2 <= {max_dz_top_v2}").groupby("T_rounded",as_index=False).median().query("T_rounded>=200").query("T_rounded<=235")

    all_cloud_iwc_median_line = all_data_whole_cloud_parameter_space.hvplot.line(x="T_rounded",y="iwc",color="black",linestyle="dashed",linewidth=3)
    all_cloud_icnc_5um_median_line = all_data_whole_cloud_parameter_space.hvplot.line(x="T_rounded",y="icnc_5um",color="black",linestyle="dashed",linewidth=3)
    all_cloud_icnc_100um_median_line = all_data_whole_cloud_parameter_space.hvplot.line(x="T_rounded",y="icnc_100um_log",color="black",linestyle="dashed",linewidth=3)
    all_cloud_reffcli_median_line = all_data_whole_cloud_parameter_space.hvplot.line(x="T_rounded",y="reffcli",color="black",linestyle="dashed",linewidth=3)

    cirrus_var = "iwc"

    iwc_median = whole_cloud_parameter_space.hvplot.line(x=f"T_rounded",y=f"{cirrus_var}_median", by=[cluster_col],height=300,width=500,legend=False, linewidth=3, color=colors,logy=logy,xlim=[200,238],xlabel="Temperature [K]",ylabel="IWC [mg m⁻³]",**plt_options)
    if cluster_col =="cluster_idx":
        iwc_median = iwc_median * all_cloud_iwc_median_line
    
    # x_dist_plt = whole_cloud_df.sort_values(cluster_col).hvplot.kde(y="T_rounded",by=cluster_col, color=colors,legend=False,alpha=0.2).opts(height=120,width=500, xaxis=None,yaxis=None)
    # y_dist_plt = whole_cloud_df.sort_values(cluster_col).hvplot.kde(y=f"{cirrus_var}_log",by=cluster_col, color=colors,legend=False,invert=True,alpha=0.2).opts(width=120,height=300, xaxis=None,yaxis=None)
    #iwc_median = iwc_median << y_dist_plt << x_dist_plt
    if quantile_lines:
        iwc_lower = whole_cloud_parameter_space.hvplot.line(x=f"T_rounded",y=f"{cirrus_var}_lower", by=[cluster_col],height=300,width=500,legend=False, linewidth=3, linestyle="dashed", color=colors,logy=logy,**plt_options)
        iwc_upper = whole_cloud_parameter_space.hvplot.line(x=f"T_rounded",y=f"{cirrus_var}_upper", by=[cluster_col],height=300,width=500,legend=False, linewidth=3, linestyle="dashed", color=colors,logy=logy,**plt_options)
        iwc = iwc_lower * iwc_upper * iwc_median
    else:
        iwc = iwc_median

    cirrus_var = "icnc_5um"

    icnc_5um_median = whole_cloud_parameter_space.hvplot.line(x=f"T_rounded",y=f"{cirrus_var}_median", by=[cluster_col],height=300,width=500,xlim=[200,238],legend=False,xlabel="Temperature [K]",ylabel="$N_{ice}$ [cm⁻³]", linewidth=3, color=colors,logy=logy,**plt_options)
    if cluster_col =="cluster_idx":
        icnc_5um_median = icnc_5um_median * all_cloud_icnc_5um_median_line
    # x_dist_plt = whole_cloud_df.sort_values(cluster_col).hvplot.kde(y="T_rounded",by=cluster_col, color=colors,legend=False,alpha=0.5).opts(height=120,width=500, xaxis=None,yaxis=None)
    # y_dist_plt = whole_cloud_df.sort_values(cluster_col).hvplot.kde(y=f"{cirrus_var}_log",by=cluster_col, color=colors,legend=False,invert=True,alpha=0.5).opts(width=120,height=300, xaxis=None,yaxis=None)
    #icnc_5um_median = icnc_5um_median << y_dist_plt << x_dist_plt
    if quantile_lines:
        icnc_5um_lower = whole_cloud_parameter_space.hvplot.line(x=f"T_rounded",y=f"{cirrus_var}_lower", by=[cluster_col],height=300,width=500,legend=False, linewidth=3, linestyle="dashed", color=colors,logy=logy,**plt_options)
        icnc_5um_upper = whole_cloud_parameter_space.hvplot.line(x=f"T_rounded",y=f"{cirrus_var}_upper", by=[cluster_col],height=300,width=500,legend=False, linewidth=3, linestyle="dashed", color=colors,logy=logy,**plt_options)
        icnc_5um = icnc_5um_lower * icnc_5um_upper * icnc_5um_median
    else:
        icnc_5um = icnc_5um_median

   
    return (iwc + icnc_5um).cols(2), whole_cloud_parameter_space 