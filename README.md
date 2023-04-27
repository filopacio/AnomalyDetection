# AnomalyDetection
Master Thesis in Statistical Sciences at University of Bologna

Title: Unsupervised Machine Learning Techniques for Anomaly Detection with Multi-Source Data
Supervisor: prof.ssa Elisabetta Ronchieri

**Data**: 
- Textual Data from Log Messages produced by different softwares and logging services running on several machines
- Numerical Data from Monitoring Process. So, different metrics representing the health status of machines were given. (e.g. CPU usage, load averages, etc.)



**Main Techniques used**:

- Natural Language Processing
- Clustering Algorithms: DBSCAN, K-Means
- Principal Component Analysis
- Time Series Outlier Detection


## Installation 

<pre><code>
git clone https://github.com/filopacio/AnomalyDetection.git
</code></pre>

## Usage

### Log
Read your Log Dataset and get a Log object

<pre><code>
df_s = pd.read_csv('YOUR PATH')
df_log = Log(df_s)
</code></pre>

This class object can plot the vectorized version of its messages, reduced in 2 or 3 components


<pre><code>
df_log.plot_2d_words_service()
#df_log.plot_3d_words_service
</code></pre>

Then use the type of Anomaly Detection object preferred

<pre><code>
db = AnomalyDbscan(df_log, reduce_dim=True, threshold=0.1, n_comps=2)
</code></pre>

And finally compute the anomaly and/or plot the results

<pre><code>
df_anom, common_anom, common_non_anom = db.compute_anomaly()
db.plot_clustering_results()
</code></pre>


### Monit








