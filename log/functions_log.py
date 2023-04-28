import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from AnomalyDetection.data.functions_data import clean_text





def compute_distance_quantile(X: np.array,
                              threshold: float):
    """
    Function that computes the 3rd quantile of a set comprising all the distances between unique different messages
    """
    distance = []
    for i in X:
        for j in X:
            if not np.array_equal(i, j):
                dist = np.linalg.norm(i - j)
                distance.append(dist)
    distance = pd.Series(distance)
    eps = np.quantile(distance, threshold)
    return eps


def compute_dbscan(df: pd.DataFrame,
                   reduce_dim: bool,
                   threshold: float,
                   n_comps: int):
    X = np.vstack(df['vec'])
    if reduce_dim:
        pca = PCA(n_components=n_comps)
        X = pca.fit_transform(X)
    # Compute DBSCAN
    epsilon = compute_distance_quantile(X, threshold)
    db = DBSCAN(eps=epsilon, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    return labels, X, core_samples_mask


def cluster_log_db(df: pd.DataFrame,
                reduce_dim: bool,
                threshold: float,
                n_comps: int):
    labels, X, core_samples_mask = compute_dbscan(df, reduce_dim, threshold, n_comps)
    df['cluster_label'] = list(labels)
    df['cluster_label'] = df['cluster_label'].apply(lambda x: 'cluster_' + str(x))
    df.loc[df['cluster_label'] == 'cluster_-1', 'anomaly'] = 1
    df.loc[df['anomaly'] != 1, 'anomaly'] = 0
    return df


def compute_anom_score(sentence: str,
                       anom_dict: dict):
    score = np.mean([anom_dict[i][0] for i in sentence])
    return score


def compute_kmeans(df, reduce_dim: bool,
                   n_comps: int,
                   n_clusters: int):
    X = np.vstack(df['vec'])
    if reduce_dim:
        pca = PCA(n_components=n_comps)
        X = pca.fit_transform(X)
    km = KMeans(n_clusters).fit(X)
    labels = km.fit_predict(X)
    return labels, X


def cluster_log_km(df: pd.DataFrame,
                   reduce_dim: bool,
                   n_comps: int,
                   n_clusters: int):
    labels, X = compute_kmeans(df, reduce_dim, n_comps, n_clusters)
    df['cluster_label_km'] = list(labels)
    df['cluster_label_km'] = df['cluster_label_km'].apply(lambda x: 'cluster_' + str(x))
    non_anomaly_label = df['cluster_label_km'].value_counts().sort_values(ascending=False).index[-1]
    df.loc[df['cluster_label_km'] == non_anomaly_label, 'anomaly'] = 0
    df.loc[df['cluster_label_km'] != non_anomaly_label, 'anomaly'] = 1
    return df


def vectorize_message(df: pd.DataFrame):
    support_df = pd.DataFrame({'msg': list(pd.unique(df_s.msg)), 'clean_msg': [clean_text(i).split() for i in list(pd.unique(df_s.msg))]})
    df_service = df_s.merge(support_df, on='msg', how='left')
    clean = list(df_service.loc[:, 'clean_msg'])
    vec = Word2Vec(clean, min_count=1)
    df_service = pd.DataFrame({'words': [i for i in vec.wv.key_to_index.keys()], 'vec': [vec.wv[i] for i in vec.wv.key_to_index.keys()]})
    #df_service = pd.DataFrame({'words': [i for i in vec.wv], 'vec': [vec[i] for i in vec.wv]})
    return df_service, support_df


def compute_pca_and_reconstruct(df: pd.DataFrame,
                                n_comps: int,
                                tolerance: float):
    X = np.vstack(df['vec'])
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
    else:
        df['anomaly'] = df['anomaly'].fillna(0)
    return df


def get_time_series_of_distances(dict_ip: dict,
                                 host: str):
    """
    Function which computes the euclidean distance between two, consecutive (in time),
    vectorized log messages. Always on a machine level.
    :param ip: string
    :return: dict -> pd.DataFrame
             distance -> a pd.Series
    """
    df = dict_ip[host].copy()
    distance = []
    for i in range(len(df) - 1):
        obs_1 = df['vec_msg'][i]
        obs_2 = df['vec_msg'][i + 1]
        dist = np.linalg.norm(obs_1 - obs_2)
        distance.append(dist)
    distance = pd.Series(distance)
    return distance


def get_anomalies_from_mean_and_sd(distance: pd.Series,
                                   tolerance: float):
    """
    Function which computes the anomalous point in a time series based on overall mean and standard deviation.
    A point is anomalous if it deviates from an interval centered on the mean and which is wide 2 times the
    tolerance, a parameter of the function, multiplied by the standard deviation

    :param: distance -> pd.Series
            tolerance -> float
    :return: anomalies -> pd.Series

    """
    anomalies = distance.loc[abs(distance) > np.mean(distance) + tolerance * np.sqrt(np.var(distance))]
    return anomalies


def label_messages_regular(dict_ip: dict, tolerance: int):
    dict_results = {}
    for host in dict_ip.keys():
        distance = get_time_series_of_distances(dict_ip, host)
        anomalies = get_anomalies_from_mean_and_sd(distance, tolerance)
        df = dict_ip[host].copy().reset_index(drop=True)
        df.loc[anomalies.index, 'anomaly_time_series'] = 1
        df.loc[df.anomaly_time_series.isna(), 'anomaly_time_series'] = 0
        dict_results[host] = df
    return dict_results


def get_anomalies_sliding_windows(distance: pd.Series,
                                  windows_size: float,
                                  tolerance: float):
    """
    Function which computes the anomalous point in a time series based on mean and standard deviation on a sliding-window
    base. A point is anomalous if it deviates from an interval centered on the mean of the observations in the window and
    which is wide 2 times the tolerance, a parameter of the function, multiplied by the window's standard deviation

    :param: distance -> pd.Series
            window_size -> int
            tolerance -> float
    :return: anomalies -> pd.Series
    """
    anomalies = [0] * len(distance)
    for i in range(len(distance) - windows_size):
        window = distance[i:i + windows_size]
        window = np.array(window)
        window_mean = np.mean(window)
        window_sd = np.sqrt(np.var(window))
        lower_limit = window_mean - tolerance * window_sd
        upper_limit = window_mean + tolerance * window_sd
        for j in window:
            if (j < lower_limit or j > upper_limit) and j not in anomalies:
                index = np.where(distance == j)[0][0]
                anomalies[index] = j
    anomalies = pd.Series(anomalies)
    anomalies = anomalies[anomalies > 0]
    return anomalies


def label_messages_sliding_windows(dict_ip: dict,
                                   windows_size: int,
                                   tolerance: int):
    """
    :param dict_ip: dictionary containing dataset for every machine
    :param windows_size: suggested values: [10,30,60,120]
    :param tolerance: suggested values: [1,2,3]
    :return: the dictionary with the results
    """
    dict_results = {}
    for host in dict_ip.keys():
        distance = get_time_series_of_distances(dict_ip, host)
        anomalies = get_anomalies_sliding_windows(distance, windows_size, tolerance)
        df = dict_ip[host].copy().reset_index(drop=True)
        df.loc[anomalies.index, 'anomaly_time_series_sliding'] = 1
        df.loc[df.anomaly_time_series_sliding.isna(), 'anomaly_time_series_sliding'] = 0
        dict_results[host] = df
    return dict_results
