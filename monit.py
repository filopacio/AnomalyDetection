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

def compute_dbscan_monit(dict_ip: dict,
                         hostname: str,
                         reduce_dim: bool,
                         n_comps: int,
                         threshold: float,
                         use_all_variables: bool = False):
    df = dict_ip[hostname].copy()
    if not use_all_variables:
        df['memory-usage'] = df['memory.used|metrics-memory'] / df['memory.total|metrics-memory']
        df = df[['time', 'iostat.avg-cpu.pct_idle|metrics-iostat-extended', 'load_avg.fifteen|metrics-load',
                 'memory-usage']]
    else:
        pass
    df = df.fillna(0)
    if not len(df) < n_comps:
        X = np.array(df.iloc[:, 1:])
        if reduce_dim:
            pca = PCA(n_components=n_comps)
            X = pca.fit_transform(X)
        X = StandardScaler().fit_transform(X)
        epsilon = threshold
        db = DBSCAN(eps=epsilon, min_samples=32).fit(X)  # min_samples = n_dim*2
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        return labels, X, core_samples_mask


def cluster_log_monit(dict_ip: dict,
                      hostname: str,
                      reduce_dim: bool,
                      n_comps: int,
                      treshold: float,
                      use_all_variables: bool):
    df = dict_ip[hostname].copy()
    if not use_all_variables:
        df['memory-usage'] = df['memory.used|metrics-memory'] / df['memory.total|metrics-memory']
        df = df[['time', 'iostat.avg-cpu.pct_idle|metrics-iostat-extended', 'load_avg.fifteen|metrics-load',
                 'memory-usage']]
    else:
        pass
    df = df.fillna(0)
    try:
        labels, X, core_samples_mask = compute_dbscan_monit(dict_ip, hostname, reduce_dim, n_comps, treshold)
        df['cluster'] = list(labels)
        df['anomaly'] = df['cluster'].apply(lambda x: '1' if x == -1 else '0')
    except TypeError:
        df['cluster'] = 'cluster_0'
        pass
    return df


def assign_label_clustering_monit(dict_ip: dict,
                                  reduce_dim: bool,
                                  n_comps: int,
                                  threshold: float,
                                  use_all_variables: bool):
    dict_results = {}
    invalid_hosts = []
    for host in dict_ip.keys():
        dict_results[host] = cluster_log_monit(dict_ip, host, reduce_dim, n_comps, threshold, use_all_variables)
        if len(pd.unique(dict_results[host].cluster)) < 2:
            invalid_hosts.append(host)
    for invalid_key in invalid_hosts:
        dict_results.pop(invalid_key)
    return dict_results


def compute_kmeans_monit(dict_ip: dict,
                         hostname: str,
                         reduce_dim: bool,
                         n_comps: int,
                         n_clusters: int,
                         use_all_variables: float):
    try:
        df = dict_ip[hostname].copy()
        if not use_all_variables:
            df['memory-usage'] = df['memory.used|metrics-memory'] / df['memory.total|metrics-memory']
            df = df[['time', 'iostat.avg-cpu.pct_idle|metrics-iostat-extended', 'load_avg.fifteen|metrics-load',
                     'memory-usage']]
        else:
            pass
        df = df.fillna(0)
        df = df.iloc[:, 1:]
        if not len(df) < n_clusters:
            scaler = StandardScaler()
            df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:].to_numpy())
            X = np.array(df.iloc[:, 1:])
            # reduce dimensionality with PCA to 2 components
            if reduce_dim:
                pca = PCA(n_components=n_comps)
                X = pca.fit_transform(X)
            km = KMeans(n_clusters).fit(X)
            labels = km.labels_
        return labels, X
    except UnboundLocalError:
        pass


def cluster_log_km_monit(dict_ip: dict,
                         host: str,
                         reduce_dim: bool,
                         n_comps: int,
                         n_clusters: int,
                         use_all_variables: float):
    try:
        df = dict_ip[host].copy()
        if not use_all_variables:
            df['memory-usage'] = df['memory.used|metrics-memory'] / df['memory.total|metrics-memory']
            df = df[['time', 'iostat.avg-cpu.pct_idle|metrics-iostat-extended', 'load_avg.fifteen|metrics-load',
                     'memory-usage']]
        else:
            pass
        df = df.fillna(0)
        labels, X = compute_kmeans_monit(dict_ip, host, reduce_dim, n_comps, n_clusters, use_all_variables)
        # add labels to dataset
        df['cluster'] = list(labels)
        anomaly_label = df['cluster'].value_counts().sort_values(ascending=False).index[-1]
        df.loc[df['cluster'] == anomaly_label, 'anomaly'] = 1
        df.loc[df['cluster'] != anomaly_label, 'anomaly'] = 0
    except TypeError:
        print('In hostname: ' + host + ' no clustering possible')
        df['cluster'] = 'cluster_0'
        pass
    return df

def assign_label_clustering_km_monit(dict_ip: dict,
                                     reduce_dim: bool,
                                     n_comps: int, n_clusters: int,
                                     use_all_variables: float):
    dict_results = {}
    invalid_hosts = []
    for host in dict_ip.keys():
        dict_results[host] = cluster_log_km_monit(dict_ip, host, reduce_dim, n_comps, n_clusters, use_all_variables)
        if len(pd.unique(dict_results[host].cluster)) < 2:
            invalid_hosts.append(host)
    for invalid_key in invalid_hosts:
        dict_results.pop(invalid_key)
    return dict_results


def compute_pca_and_reconstruct_monit(dict_ip: dict,
                                      host: str,
                                      n_comps: int,
                                      tolerance: float,
                                      use_all_variables: bool):
    df = dict_ip[host].copy()
    if not use_all_variables:
        df['memory-usage'] = df['memory.used|metrics-memory'] / df['memory.total|metrics-memory']
        df = df[['time', 'iostat.avg-cpu.pct_idle|metrics-iostat-extended', 'load_avg.fifteen|metrics-load',
                 'memory-usage']]
    else:
        pass
    df_1 = df.fillna(0)
    scaler = StandardScaler()
    if not df_1.empty:
        df_1.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:].to_numpy())
        X = np.array(df_1.iloc[:, 1:])
        X = np.nan_to_num(X)
        pca = PCA().fit(X)
        trans_x = pca.transform(X)
        p_comp = pca.components_
        result = np.dot(trans_x[:, 0:n_comps], p_comp[0:n_comps, :])
        diff = X - result
        diff_sq = diff * diff
        errs = np.sum(diff_sq, axis=1)
        for i in np.argwhere(errs > np.quantile(errs, tolerance)):
            df.loc[i, 'anomaly'] = 1
        if 'anomaly' not in df.columns:
            print('No anomaly detected for host: ' + host)
            df['anomaly'] = 0
        else:
            df['anomaly'] = df['anomaly'].fillna(0)
    else:
        df = pd.DataFrame()
        print(host, ' has empty dataset, no anomaly detection possible')
    return df


def assign_label_pca_monit(dict_ip: dict,
                           n_comps: int,
                           tolerance: float,
                           use_all_variables: bool):
    dict_results = {}
    invalid_hosts = []
    for host in dict_ip.keys():
        dict_results[host] = compute_pca_and_reconstruct_monit(dict_ip, host, n_comps, tolerance, use_all_variables)
        if dict_results[host].empty:
            invalid_hosts.append(host)
    for invalid_key in invalid_hosts:
        dict_results.pop(invalid_key)
    return dict_results


def get_time_series_of_distances_monit(dict_ip: dict,
                                       host: str,
                                       use_all_variables: bool):
    """
    Function which computes the euclidean distance between two, consecutive (in time),
    vectorizedlog messages. Always on a machine level.

    :param ip: string
    :return: dict -> pd.DataFrame
             distance -> a pd.Series
    """
    df = dict_ip[host].copy()
    distance = []
    if not use_all_variables:
        df['memory-usage'] = df['memory.used|metrics-memory'] / df['memory.total|metrics-memory']
        df = df[['time', 'iostat.avg-cpu.pct_idle|metrics-iostat-extended', 'load_avg.fifteen|metrics-load',
                 'memory-usage']]
    else:
        pass
    df = df.fillna(0)
    X = df.iloc[:, 2:].to_numpy()
    for i in range(len(X) - 1):
        obs_1 = X[i]
        obs_2 = X[i + 1]
        dist = np.linalg.norm(obs_1 - obs_2)
        distance.append(dist)
    distance = pd.Series(distance)
    return distance, df


def label_messages_regular_monit(dict_ip: dict, monit_type: str, tolerance, use_all_variables: bool):
    " monit_type can be either 'storage' or 'farming' "
    invalid_hosts = []
    dict_results = {}
    for host in dict_ip.keys():
        distance, df = get_time_series_of_distances_monit(dict_ip, host, monit_type, use_all_variables)
        anomalies = get_anomalies_from_mean_and_sd(distance, tolerance)
        if not df.empty:
            df.loc[anomalies.index, 'anomaly_time_series'] = 1
            df.loc[df.anomaly_time_series.isna(), 'anomaly_time_series'] = 0
            dict_results[host] = df
        else:
            print('Hostname: ' + host + ' is empty, no time series analysis possible')
            invalid_hosts.append(host)
    for i in invalid_hosts:
        dict_ip.pop(i)
    return dict_results


def label_messages_sliding_monit(dict_ip: dict, windows_size, tolerance, use_all_variables: bool):
    invalid_hosts = []
    dict_results = {}
    for host in dict_ip.keys():
        distance, df = get_time_series_of_distances_monit(dict_ip, host, use_all_variables)
        anomalies = get_anomalies_sliding_windows(distance, windows_size, tolerance)
        if not df.empty:
            df.loc[anomalies.index, 'anomaly_sliding'] = 1
            df.loc[df.anomaly_sliding.isna(), 'anomaly_sliding'] = 0
            dict_results[host] = df
        else:
            print('Hostname: ' + host + ' is empty, no time series analysis possible')
            invalid_hosts.append(host)
    for i in invalid_hosts:
        dict_ip.pop(i)
    return dict_results


def plot_distance_and_anomaly(distance, anomalies, sliding_windows):
    """
    Function to plot a time series and his anomalous points

    :param: distance -> pd.Series
            anomalies -> pd.Series
    :return: plot
    """
    ax = distance.plot()
    anomalies.plot(marker='o', ls='', ax=ax, color='r')
    if sliding_windows:
        plt.title('Anomaly Detection with sliding windows')
        plt.xlabel('time')
        plt.ylabel('distance')
    else:
        plt.title('Regular Anomaly Detection')
    plt.show()


def plot_distance_and_anomalies_from_host_storage(host, windows_size, tolerance, sliding_windows):
    distance = get_time_series_of_distances_monit(host, 'storage')
    if not sliding_windows:
        anomaly = get_anomalies_from_mean_and_sd(distance, tolerance)
    else:
        anomaly = get_anomalies_sliding_windows(distance, windows_size, tolerance)
    plot_distance_and_anomaly(distance, anomaly, sliding_windows)


def plot_distance_and_anomalies_from_host_farming(host, windows_size, tolerance, sliding_windows):
    distance = get_time_series_of_distances_monit(host, 'farming')
    if not sliding_windows:
        anomaly = get_anomalies_from_mean_and_sd(distance, tolerance)
    else:
        anomaly = get_anomalies_sliding_windows(distance, windows_size, tolerance)
    plot_distance_and_anomaly(distance, anomaly, sliding_windows)


def hypothesis_monit(dict_results: dict):
    metrics = []
    dict_metrics_anom = {}
    dict_metrics_non_anom = {}
    for k in dict_results.keys():
        dict_metrics_anom[k] = {}
        dict_metrics_non_anom[k] = {}
        for data in dict_results[k].keys():
            df = dict_results[k][data]
            for metric in df.columns[1:-2]:
                dict_metrics_anom[k][metric] = []
                dict_metrics_non_anom[k][metric] = []
                non_anom = df[df.anomaly == '0'][metric].fillna(0)
                anom = df[df.anomaly == '1'][metric].fillna(0)
                dict_metrics_anom[k][metric].append(np.mean(anom))
                dict_metrics_non_anom[k][metric].append(np.mean(non_anom))
                if stats.ranksums(non_anom, anom)[1] < 0.05:
                    metrics.append(metric)
    count_anom_metrics = Counter(metrics)
    most_recurring_metrics = count_anom_metrics.most_common(18)
    return dict_metrics_non_anom, dict_metrics_anom, most_recurring_metrics


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