import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from functions_log import get_anomalies_from_mean_and_sd, get_anomalies_sliding_windows
from functions_monit import *


class Monit():
    def __init__(self,
                 monit_type: str,
                 dict_ip: dict):
        self.monit_type = monit_type
        self.dict_ip = dict_ip

    def list_hosts_dict(self):
        dict_ip = self.dict_ip
        list_keys = list(dict_ip.keys())
        return list_keys


class AnomalyDbscan(Monit):
    def __init__(self,
                 monit_type: str,
                 dict_ip: dict,
                 reduce_dim: bool,
                 n_comps: int,
                 threshold: float,
                 use_all_variables: bool):

        self.monit_type = monit_type
        self.dict_ip = dict_ip
        self.reduce_dim = reduce_dim
        self.n_comps = n_comps
        self.threshold = threshold
        self.use_all_variables = use_all_variables
        super().__init__(monit_type, dict_ip)

    def compute_anomaly(self):
        dict_ip = self.dict_ip
        reduce_dim = self.reduce_dim
        n_comps = self.n_comps
        threshold = self.threshold
        use_all_variables = self.use_all_variables
        dict_results = assign_label_clustering_monit(dict_ip=dict_ip,
                                                     reduce_dim=reduce_dim,
                                                     n_comps=n_comps,
                                                     threshold=threshold,
                                                     use_all_variables=use_all_variables)
        return dict_results


class AnomalyKMeans(Monit):
    def __init__(self,
                 monit_type: str,
                 dict_ip: dict,
                 reduce_dim: bool,
                 n_comps: int,
                 n_clusters: int,
                 use_all_variables: bool):

        self.dict_ip = dict_ip
        self.monit_type = monit_type
        self.reduce_dim = reduce_dim
        self.n_comps = n_comps
        self.n_clusters = n_clusters
        self.use_all_variables = use_all_variables
        super().__init__(monit_type, dict_ip)

    def compute_anomaly(self):
        dict_ip = self.dict_ip
        reduce_dim = self.reduce_dim
        n_comps = self.n_comps
        n_clusters = self.n_clusters
        use_all_variables = self.use_all_variables
        dict_results = assign_label_clustering_km_monit(dict_ip=dict_ip,
                                                        reduce_dim=reduce_dim,
                                                        n_comps=n_comps,
                                                        n_clusters=n_clusters,
                                                        use_all_variables=use_all_variables)
        return dict_results


class AnomalyPCA(Monit):
    def __init__(self,
                 monit_type: str,
                 dict_ip: dict,
                 n_comps: int,
                 tolerance: float,
                 use_all_variables: bool):

        self.dict_ip = dict_ip
        self.monit_type = monit_type
        self.n_comps = n_comps
        self.tolerance = tolerance
        self.use_all_variables = use_all_variables
        super().__init__(monit_type, dict_ip)

    def compute_anomaly(self):
        dict_ip = self.dict_ip
        tolerance = self.tolerance
        n_comps = self.n_comps
        use_all_variables = self.use_all_variables
        dict_results = assign_label_pca_monit(dict_ip=dict_ip,
                                              n_comps=n_comps,
                                              tolerance=tolerance,
                                              use_all_variables=use_all_variables)
        return dict_results


class AnomalyTimeSeries(Monit):
    def __init__(self,
                 monit_type: str,
                 dict_ip: dict,
                 sliding_windows: bool,
                 windows_size: int,
                 tolerance: float,
                 use_all_variables: bool):

        self.dict_ip = dict_ip
        self.monit_type = monit_type
        self.sliding_windows = sliding_windows
        self.windows_size = windows_size
        self.tolerance = tolerance
        self.use_all_variables = use_all_variables
        super().__init__(monit_type, dict_ip)

    def compute_anomaly(self):
        dict_ip = self.dict_ip
        sliding_windows = self.sliding_windows
        windows_size = self.windows_size
        tolerance = self.tolerance
        use_all_variables = self.use_all_variables
        if not sliding_windows:
            dict_results = label_messages_regular_monit(dict_ip=dict_ip,
                                                        tolerance=tolerance,
                                                        use_all_variables=use_all_variables)
        else:
            dict_results = label_messages_sliding_monit(dict_ip=dict_ip,
                                                        windows_size=windows_size,
                                                        tolerance=tolerance,
                                                        use_all_variables=use_all_variables)
        return dict_results

    def plot_distance_and_anomaly(self,
                                  host: str):
        sliding_windows = self.sliding_windows
        tolerance = self.tolerance
        windows_size = self.windows_size
        monit_type = self.monit_type
        use_all_variables = self.use_all_variables
        if monit_type == 'farming':
            plot_distance_and_anomalies_from_host_farming(host=host,
                                                          windows_size=windows_size,
                                                          tolerance=tolerance,
                                                          sliding_windows=sliding_windows,
                                                          use_all_variables=use_all_variables)
        else:
            plot_distance_and_anomalies_from_host_storage(host=host,
                                                          windows_size=windows_size,
                                                          tolerance=tolerance,
                                                          sliding_windows=sliding_windows,
                                                          use_all_variables=use_all_variables)
