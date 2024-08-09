import numpy as np
import pandas as pd

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def sel_cloud_center(x: pd.DataFrame):
    """select cloud center
    
    Args:
        x: df of single cloud
    Return:
        df with single row representing the center layer of the cloud
    """
    
    # return cloud base in case cloud has 1 or 2 layers
    if len(x) <= 2:
        return x.iloc[[-1]]
    # if even number of cloud layers, return center layer closer to cloud base
    elif len(x) % 2 == 0:
        return x.iloc[[-len(x)/2]]
    # if uneven number of cloud layers, return center layer
    else:
        assert len(x) % 2 == 1
        return x.iloc[[-np.ceil(len(x)/2)]]
