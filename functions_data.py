import pandas as pd
import glob
import numpy as np
import re
import warnings
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
import wordcloud
import functools

PATH_LOG = ""  # to be filled
filenames_log = glob.glob(PATH_LOG + "/*.csv")

PATH_STORE = "" # to be filled
filenames_storage = glob.glob(PATH_STORE + "/*.csv")
PATH_FARM = "" # to be filled
filenames_farming = glob.glob(PATH_FARM + "/*.csv")


def clean_text(text_in: str) -> str:
    """
    Takes a string and returns the string with all the words in lower text and without digits
    :param text_in: the string to be cleaned
    :return text_out: the string cleaned
    """
    text = str(text_in).lower()  # all in lower case
    text = re.sub("[\d-]", ' ', text)  # to remove digits
    text = re.sub("https://\S+|www\.\S", ' ', text)  # to remove link
    text_out = re.sub('\W+', ' ', text)  # to remove punctuation and special characters
    return text_out


def get_dataframes_per_host(filenames: list,
                            indexes_to_use: list = [[0, 11, 12, 15, 16, 20, 30, 31, 33, 35, 38]]) -> dict:
    """
    Takes the path of the repository containing the csv file and returns a dictionary with one dataset for every ip (or network)

    :param path: a string with the general path for the repository where datafiles are stored
    :return dict_ip: dictionary with a dataframe of log events for every ip address (so for every machine)
    """
    if not indexes_to_use == 'all':
        filenames = [filenames[i] for i in indexes_to_use]
    chunksize = 10 ** 6
    dict_ip = {}
    for filename in filenames:
        for chunk in pd.read_csv(filename, chunksize=chunksize, low_memory=False):
            df = chunk.dropna()
            df = df[['timestamp', 'hostname', 'ip', 'process_name', 'msg']]
            df['clean_msg'] = df['msg'].apply(clean_text)
            if list(pd.unique(df['hostname'])):
                for hostname in list(pd.unique(df['hostname'])):
                    if hostname not in dict_ip.keys():
                        dict_ip[hostname] = df[df['hostname'] == hostname].reset_index(drop=True)
                    else:
                        dict_ip[hostname] = dict_ip[hostname].append(df[df['hostname'] == hostname])
                    dict_ip[hostname] = dict_ip[hostname].sort_values('timestamp').reset_index(drop=True)
    return dict_ip


def vectorize_msg(df):
    df['clean_msg'] = df['msg'].apply(clean_text)
    df['clean_msg'] = df['clean_msg'].apply(lambda x: x.split())
    # vectorize
    clean = list(df.loc[:, 'clean_msg'])
    vec = Word2Vec(clean, min_count=1)
    df['vec_msg'] = df.loc[:, 'clean_msg'].apply(lambda x: vectorize_sentence_from_single_vectors(x, vec))
    return df


def vectorize_log(dict_ip: dict) -> dict:
    """
    Takes the dictionary containing the dataframes for every ip and returns a dictionary where the log messages
    are vectorized with Word2Vec model, for every ip.

    :param dict_ip: a dictionary
    :return msg_w2v_dict: a dictionary
    """
    for host in dict_ip.keys():
        df = dict_ip[host]
        df = vectorize_msg(df)
        dict_ip[host] = df
    return dict_ip


def get_log_dataframes_per_host(filenames: list, indexes_to_use: list) -> dict:
    dict_ip_log = get_log_dataframes_per_host(filenames, indexes_to_use)
    dict_ip_log = vectorize_log(dict_ip_log)
    return dict_ip_log


def get_monit_dataframe_per_host(filenames: list) -> dict:
    dict_ip_monit = {}
    for filename in filenames:
        host = filename[107:-4]
        dict_ip_monit[host] = pd.read_csv(filename, low_memory=False).drop('Unnamed: 0', axis=1)
    return dict_ip_monit
