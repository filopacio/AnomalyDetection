import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from collections import Counter
import wordcloud
from data import clean_text
from functions_log import *


def clean_text(text_in: str) -> str:
    """
    Takes a string and returns the string with all the words in lower text and without digits
    :param text_in: the string to be cleaned
    :return text_out: the string cleaned
    """
    text = str(text_in).lower()  # all in lower case
    text = re.sub("[\d-]", ' ', text)  # to remove digits
    text = re.sub('https://\S+|www\.\S', ' ', text)  # to remove link
    text_out = re.sub('\W+', ' ', text)  # to remove punctuation and special characters
    return text_out


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


def cluster_log(df: pd.DataFrame,
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
    support_df = pd.DataFrame(
        {'msg': list(pd.unique(df.msg)), 'clean_msg': [clean_text(i).split() for i in list(pd.unique(df.msg))]})
    df_service = df.merge(support_df, on='msg', how='left')
    clean = list(df_service.loc[:, 'clean_msg'])
    vec = Word2Vec(clean, min_count=1)
    df_service = pd.DataFrame({'words': [i for i in vec.wv.vocab], 'vec': [vec[i] for i in vec.wv.vocab]})
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


class Log():
    def __init__(self,
                 df: pd.DataFrame):
        self.df = df

    def plot_2d_words_service(self):
        df = self.df
        df = vectorize_message(df)[0]
        X = np.vstack(df['vec'])
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        plt.figure(figsize=(15, 15))
        plt.scatter(result[:, 0], result[:, 1])
        plt.rcParams.update({'font.size': 25})
        words = list([i for i in df['words'] if len(i) > 3])
        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        plt.xlabel('comp. 1')
        plt.ylabel('comp. 2')
        plt.grid()
        plt.show()

    def plot_3d_words_service(self):
        df = self.df
        df = vectorize_message(df)[0]
        X = np.vstack(df['vec'])
        pca = PCA(n_components=3)
        result = pca.fit_transform(X)
        x, y, z = [], [], []
        plt.rcParams.update({'font.size': 22})
        for i in range(len(result)):
            x.append(result[i, 0])
            y.append(result[i, 1])
            z.append(result[i, 2])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z)
        ax.set_xlabel('comp. 1')
        ax.set_ylabel('comp. 2')
        ax.set_zlabel('comp. 3')
        for x, y, z, label in zip(x, y, z):
            ax.text(x, y, z, label)
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
        plt.show()


class AnomalyDbscan(Log):
    def __init__(self,
                 df: pd.DataFrame,
                 reduce_dim: bool,
                 threshold: float,
                 n_comps: int):

        self.df = df
        self.reduce_dim = reduce_dim
        self.threshold = threshold
        self.n_comps = n_comps
        super().__init__(df)

    def compute_anomaly(self):
        df_service = self.df
        reduce_dim = self.reduce_dim
        threshold = self.threshold
        n_comps = self.n_comps
        number_anom_words = 0
        number_non_anom_words = 0
        len_anom_msg = []
        len_non_anom_msg = []
        df, support_df = vectorize_message(df_service)
        df_temp = cluster_log(df, reduce_dim, threshold, n_comps)
        anom_dict = df_temp[['words', 'anomaly']].groupby('words')['anomaly'].apply(list).to_dict()
        support_df['anom_score'] = support_df['clean_msg'].apply(lambda x: compute_anom_score(x, anom_dict))
        df_service_temp = df_service.merge(support_df, on='msg')
        anomal_msg = list(df_service_temp[df_service_temp['anom_score'] > 0.95].clean_msg)
        non_anomal_msg = list(df_service_temp[df_service_temp['anom_score'] < 0.1].clean_msg)
        anomaly = [i for y in anomal_msg for i in y]
        non_anomaly = [i for y in non_anomal_msg for i in y]
        counts_anom = Counter([i for i in anomaly if len(i) > 3])
        counts_non_anom = Counter([i for i in non_anomaly if len(i) > 3])
        max_anom = counts_anom.most_common(10)
        max_non_anom = counts_non_anom.most_common(10)
        common_anom_words = [i[0] for i in max_anom]
        common_non_anom_words = [i[0] for i in max_non_anom]
        number_anom_words += len(df_temp[df_temp.anomaly == 1])
        number_non_anom_words += len(df_temp[df_temp.anomaly == 0])
        len_anom_msg.append(len(anomal_msg))
        len_non_anom_msg.append(len(non_anomal_msg))
        return df_service_temp, common_anom_words, common_non_anom_words

    def plot_clustering_results(self):
        df_service = self.df
        reduce_dim = self.reduce_dim
        threshold = self.threshold
        n_comps = self.n_comps
        df, support_df = vectorize_message(df_service)
        labels, X, core_mask = compute_dbscan(df=df, reduce_dim=reduce_dim, n_comps=n_comps, threshold=threshold)
        filtered_label0 = X[labels != -1]
        filtered_label1 = X[labels == -1]
        words0 = df.words[labels != -1]
        words1 = df.words[labels == -1]
        plt.figure(figsize=(15, 15))
        plt.rcParams.update({'font.size': 23})
        plt.xlabel('comp.1')
        plt.ylabel('comp.2')
        plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
        for i, word in enumerate(words0):
            plt.annotate(word, xy=(filtered_label0[i, 0], filtered_label0[i, 1]))
        plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1], color='red')
        for i, word in enumerate(words1):
            plt.annotate(word, xy=(filtered_label1[i, 0], filtered_label1[i, 1]))
        plt.legend(['regular', 'anomalous'], loc='upper left')
        plt.grid()
        plt.show()


class AnomalyKmeans(Log):
    def __init__(self,
                 df: pd.DataFrame,
                 reduce_dim: bool,
                 n_clusters: float,
                 n_comps: int):

        self.df = df
        self.reduce_dim = reduce_dim
        self.n_clusters = n_clusters
        self.n_comps = n_comps
        super().__init__(df)

    def compute_anomaly(self):
        df_service = self.df
        reduce_dim = self.reduce_dim
        n_clusters = self.n_clusters
        n_comps = self.n_comps
        number_anom_words = 0
        number_non_anom_words = 0
        len_anom_msg = []
        len_non_anom_msg = []
        df, support_df = vectorize_message(df_service)
        df_temp = cluster_log_km(df, reduce_dim, n_comps, n_clusters)
        anom_dict = df_temp[['words', 'anomaly']].groupby('words')['anomaly'].apply(list).to_dict()
        support_df['anom_score'] = support_df['clean_msg'].apply(lambda x: compute_anom_score(x, anom_dict))
        df_service_temp = df_service.merge(support_df, on='msg')
        anomal_msg = list(df_service_temp[df_service_temp['anom_score'] > 0.8].clean_msg)
        non_anomal_msg = list(df_service_temp[df_service_temp['anom_score'] < 0.2].clean_msg)
        anomaly = [i for y in anomal_msg for i in y]
        non_anomaly = [i for y in non_anomal_msg for i in y]
        counts_anom = Counter([i for i in anomaly if len(i) > 3])
        counts_non_anom = Counter([i for i in non_anomaly if len(i) > 3])
        max_anom = counts_anom.most_common(10)
        max_non_anom = counts_non_anom.most_common(10)
        common_anom_words = [i[0] for i in max_anom]
        common_non_anom_words = [i[0] for i in max_non_anom]
        number_anom_words += len(df_temp[df_temp.anomaly == 1])
        number_non_anom_words += len(df_temp[df_temp.anomaly == 0])
        len_anom_msg.append(len(anomal_msg))
        len_non_anom_msg.append(len(non_anomal_msg))
        return df_service_temp, common_anom_words, common_non_anom_words

    def plot_clustering_results(self):
        df_service = self.df
        reduce_dim = self.reduce_dim
        n_clusters = self.n_clusters
        n_comps = self.n_comps
        df, support_df = vectorize_message(df_service)
        labels, X = compute_kmeans(df=df, reduce_dim=reduce_dim, n_comps=n_comps, n_clusters=n_clusters)
        filtered_label0 = X[labels == 0]
        filtered_label1 = X[labels == 1]
        filtered_label2 = X[labels == 2]
        words0 = df.words[labels == 0]
        words1 = df.words[labels == 1]
        words2 = df.words[labels == 2]
        plt.figure(figsize=(15, 15))
        plt.rcParams.update({'font.size': 23})
        plt.xlabel('comp.1')
        plt.ylabel('comp.2')
        plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
        for i, word in enumerate(words0):
            plt.annotate(word, xy=(filtered_label0[i, 0], filtered_label0[i, 1]))
        plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1])
        for i, word in enumerate(words1):
            plt.annotate(word, xy=(filtered_label1[i, 0], filtered_label1[i, 1]))
        plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1])
        for i, word in enumerate(words2):
            plt.annotate(word, xy=(filtered_label2[i, 0], filtered_label2[i, 1]))
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()


class AnomalyPCA(Log):
    def __init__(self,
                 df: pd.DataFrame,
                 tolerance: float,
                 n_comps: int):
        self.df = df
        self.tolerance = tolerance
        self.n_comps = n_comps
        super().__init__(df)

    def compute_anomaly(self):
        df_service = self.df
        tolerance = self.tolerance
        n_comps = self.n_comps
        number_anom_words = 0
        number_non_anom_words = 0
        len_anom_msg = []
        len_non_anom_msg = []
        df, support_df = vectorize_message(df_service)
        df_temp = compute_pca_and_reconstruct(df, n_comps, tolerance)
        anom_dict = df_temp[['words', 'anomaly']].groupby('words')['anomaly'].apply(list).to_dict()
        support_df['anom_score'] = support_df['clean_msg'].apply(lambda x: compute_anom_score(x, anom_dict))
        df_service_temp = df_service.merge(support_df, on='msg')
        anomal_msg = list(df_service_temp[df_service_temp['anom_score'] > 0.95].clean_msg)
        non_anomal_msg = list(df_service_temp[df_service_temp['anom_score'] < 0.1].clean_msg)
        anomaly = [i for y in anomal_msg for i in y]
        non_anomaly = [i for y in non_anomal_msg for i in y]
        counts_anom = Counter([i for i in anomaly if len(i) > 3])
        counts_non_anom = Counter([i for i in non_anomaly if len(i) > 3])
        max_anom = counts_anom.most_common(10)
        max_non_anom = counts_non_anom.most_common(10)
        common_anom_words = [i[0] for i in max_anom]
        common_non_anom_words = [i[0] for i in max_non_anom]
        number_anom_words += len(df_temp[df_temp.anomaly == 1])
        number_non_anom_words += len(df_temp[df_temp.anomaly == 0])
        len_anom_msg.append(len(anomal_msg))
        len_non_anom_msg.append(len(non_anomal_msg))
        return df_service_temp, common_anom_words, common_non_anom_words


class LogAnomalyTimeSeries():
    def __init__(self,
                 dict_ip: dict,
                 tolerance: int,
                 sliding_windows: bool,
                 windows_size: int = 30):

        self.dict_ip = dict_ip
        self.sliding_windows = sliding_windows
        self.windows_size = windows_size
        self.tolerance = tolerance

    def host_list(self):
        dict_ip = self.dict_ip
        keys_list = list(dict_ip.keys())
        return keys_list

    def compute_anomaly(self):
        dict_ip = self.dict_ip
        sliding_windows = self.sliding_windows
        windows_size = self.windows_size
        tolerance = self.tolerance
        if not sliding_windows:
            dict_results = label_messages_regular(dict_ip=dict_ip, tolerance=tolerance)
        else:
            dict_results = label_messages_sliding_windows(dict_ip, windows_size=windows_size, tolerance=tolerance)
        return dict_results

    def plot_distance_and_anomaly(self,
                                  host: str):
        """
        Function to plot a time series and his anomalous points

        :param: distance -> pd.Series
                anomalies -> pd.Series
        :return: plt.plot
        """
        dict_ip = self.dict_ip
        sliding_windows = self.sliding_windows
        windows_size = self.windows_size
        tolerance = self.tolerance
        distance = get_time_series_of_distances(dict_ip=dict_ip, host=host)
        ax = distance.plot()
        plt.xlabel('time')
        plt.ylabel('distance')
        if sliding_windows:
            anomalies = get_anomalies_from_mean_and_sd(distance=distance, tolerance=tolerance)
            anomalies.plot(marker='o', ls='', ax=ax, color='r')
            plt.title('Anomaly Detection with sliding windows')
            plt.show()
        else:
            anomalies = get_anomalies_sliding_windows(distance=distance, windows_size=windows_size, tolerance=tolerance)
            anomalies.plot(marker='o', ls='', ax=ax, color='r')
            plt.title('Regular Anomaly Detection')
            plt.show()


def plot_wordcloud(df_anom: pd.DataFrame):
    list_msg = df_anom[df_anom.anom_score < 0.3].clean_msg
    wc = wordcloud.WordCloud(collocations=False, background_color='black', max_words=100, max_font_size=50,
                             stopwords=['dict_keys', '-'])
    wc = wc.generate(','.join([y for i in list_msg for y in i if len(y) > 3]))
    plt.axis('off')
    plt.imshow(wc)
    plt.title('Wordcloud for Service' + df_anom['process_name'][0], 'Anomaly Score < 0.3')
    plt.show()
    list_msg = df_anom[df_anom.anom_score > 0.7].clean_msg
    wc = wordcloud.WordCloud(collocations=False, background_color='black', max_words=10, max_font_size=50,
                             stopwords=['dict_keys', '-'])
    wc = wc.generate(','.join([y for i in list_msg for y in i if len(y) > 3]))
    plt.axis('off')
    plt.imshow(wc)
    plt.title('Wordcloud for Service' + df_anom['process_name'][0], 'Anomaly Score > 0.7')
    plt.show()