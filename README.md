# Anomaly Detection in Data Center IT Infrastructure using Natural Language Processing, PCA Reconstruction Error and Time Series Solutions

Master Thesis in Statistical Sciences at University of Bologna

Supervisor: prof.ssa Elisabetta Ronchieri

Title: Unsupervised Machine Learning Techniques for Anomaly Detection with Multi-Source Data

Publication: [Here's the link to the paper based on the thesis](https://pos.sissa.it/434/024/pdf)




**Data**: 
- Textual Data from Log Messages produced by different softwares and logging services running on several machines
- Numerical Data from Monitoring Process. So, different metrics representing the health status of machines were given. (e.g. CPU usage, load averages, etc.)


**Main Techniques used**:

- Natural Language Processing
- Clustering Algorithms: DBSCAN, K-Means
- Principal Component Analysis
- Time Series Outlier Detection


## Installation 

```sh
git clone https://github.com/filopacio/AnomalyDetection.git
```

## Usage

### Log
Read your Log Dataset and get a Log object

```sh
df_s = pd.read_csv('YOUR PATH')
df_log = Log(df_s)
```

This class object can plot the vectorized version of its messages, reduced in 2 or 3 components


```sh
df_log.plot_2d_words_service()
df_log.plot_3d_words_service()
```

![alt text](https://github.com/filopacio/AnomalyDetection/blob/main/images/2d.png)

![alt text](https://github.com/filopacio/AnomalyDetection/blob/main/images/3d.png)

Then use the type of Anomaly Detection object preferred

```sh
db = AnomalyDbscan(df_log, reduce_dim=True, threshold=0.1, n_comps=2)
```

And finally compute the anomaly and/or plot the results

```sh
df_anom, common_anom, common_non_anom = db.compute_anomaly()
db.plot_clustering_results()
```
![alt text](https://github.com/filopacio/AnomalyDetection/blob/main/images/db.png)

### Monit

```sh
ts_monit_storage = MonitAnomalyTimeSeries(monit_type='farming', 
                                          dict_ip=dict_farm, 
                                          sliding_windows=False, 
                                          windows_size=35,
                                          tolerance=0.9, 
                                          use_all_variables=True)
```

And then plot the results for a particular machine
```sh
ts_monit_storage.plot_distance_and_anomaly(host='farm_dbfarm-2')
```
![alt text](https://github.com/filopacio/AnomalyDetection/blob/main/images/ts_monit.png)







