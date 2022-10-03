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
from functions_data import clean_text
from functions_log import *

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


class LogAnomalyDbscan(Log):
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


class LogAnomalyKmeans(Log):
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


class LogAnomalyPCA(Log):
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
    
    def plot_recon_err_pca(self):
       df_service = self.df
       tolerance = self.tolerance
       n_comps = self.n_comps
       df = dict_ip_store[host].copy()
       df = df.fillna(0)
       scaler = StandardScaler()
       df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:].to_numpy())
       if not df.empty:
          X = np.array(df_1.iloc[:, 1:])
          pca = PCA().fit(X)
          trans_x = pca.transform(X)
          p_comp = pca.components_
          result = np.dot(trans_x[:, 0:n_comps], p_comp[0:n_comps, :])
          diff = X - result
          diff_sq = diff * diff
          errs = np.sum(diff_sq, axis=1)
       plt.figure(figsize=(15,5))
       ax = errs.plot()
       anomalies.plot(marker='o', ls='', ax=ax, color='r')
       plt.xlabel('index')
       plt.ylabel('recon. error')
       plt.legend(['series', 'anomalies'])
       plt.grid()
       plt.show()


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
