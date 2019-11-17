import pyspark
import matplotlib.pyplot as plt
from pyspark.sql import functions
import os
import sys
import findspark
import time
import json
import numpy as np
import operator
import jsonlines
import pandas as pd
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.clustering import KMeans, GaussianMixture
import pyspark.ml.feature as feat
findspark.init()
spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("Python Spark K-means minimal example")
    .enableHiveSupport()
    .getOrCreate()
)
path_aggregated_df=""
path_metrics_kmeans_sse=""
# cia keliai skaitomi is json failo, jau dirbame su parquet failu
json_file_path = "Params.json"
with open(json_file_path, 'r') as j:
     contents = json.load(j)
cluster=contents['cluster']
for item in cluster:
    path_aggregated_df=item['path_aggregated_df']
    path_metrics_kmeans_sse=item['path_metrics_kmeans_sse']
#path_aggregated_df = "../data/sample_aggregated_usage_with_churn"
clustering_df = spark.read.parquet(path_aggregated_df)
# destytojas sake kad reikai ismesti dali stulpeliu, na teorishkai reikia nuspresti kurie reiksmingi,
# tame ismesti, turbut butu gerai imesti i json'a, kad galima butu koreguoti nekeiciant kodo
columns_clustering_features = [
    'user_lifetime',
    'user_no_outgoing_activity_in_days',
    'user_account_balance_last',
    'user_spendings',
    'reloads_inactive_days',
    'reloads_sum',
    'calls_outgoing_count',
    'calls_outgoing_spendings',
    'calls_outgoing_to_abroad_spendings',
    'calls_outgoing_to_abroad_duration',
    'calls_outgoing_to_abroad_inactive_days',
    'sms_outgoing_count',
    'sms_outgoing_spendings',
    'sms_outgoing_spendings_max',
    'sms_outgoing_to_onnet_spendings',
    'sms_outgoing_to_onnet_inactive_days',
    'sms_outgoing_to_offnet_count',
    'sms_outgoing_to_offnet_spendings',
    'sms_outgoing_to_offnet_inactive_days',
    'sms_outgoing_to_abroad_count',
    'sms_outgoing_to_abroad_spendings',
    'sms_outgoing_to_abroad_inactive_days',
    'sms_incoming_count',
    'sms_incoming_spendings',
    'sms_incoming_from_abroad_count',
    'sms_incoming_from_abroad_spendings',
    'gprs_session_count',
    'gprs_usage',
    'gprs_spendings',
    'gprs_inactive_days',
    'last_100_reloads_count',
    'last_100_reloads_sum',
 
]
# duomenu paruosimas
vector_assembler = VectorAssembler(
    inputCols=columns_clustering_features, 
    outputCol="initial_features")

#selector = feat.ChiSqSelector(
#labelCol='churn'
#, numTopFeatures=10
#, outputCol='selected')
#pipeline_sel = Pipeline(stages=[vector_assembler, selector])

#result =(
#pipeline_sel
#.fit(clustering_df)
#.transform(clustering_df)

#)
#print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
#result.show()
# standartizavimas
standard_scaler = StandardScaler(
    inputCol="initial_features", 
    outputCol="features", 
    withStd=True, 
    withMean=True)
vectorized_df = vector_assembler.transform(clustering_df)
model_scaler = standard_scaler.fit(vectorized_df)
featurized_clustering_df = model_scaler.transform(vectorized_df)
featurization_pipeline = Pipeline(stages=[vector_assembler, standard_scaler])
featurization_pipeline_model = featurization_pipeline.fit(clustering_df)
model_scaler = featurization_pipeline_model.stages[-1]
featurized_clustering_df = featurization_pipeline_model.transform(clustering_df)
sse_cost = np.zeros(20)
#path_metrics_kmeans_sse = "../data/metrics_kmeans_see.jsonl"
# pradedu klasteriu parinkima metrikas saugau json faile, kartu sukuriau image faila
# kuriame nupiesta kreive kurios pagalba rasime kiek klasteriu reikia
for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(featurized_clustering_df.sample(False,0.1, seed=42))
    sse_cost[k] = model.computeCost(featurized_clustering_df)
    metrics_row = {"k": k, "sse": sse_cost[k]}
    # metrikas i json
    with jsonlines.open(path_metrics_kmeans_sse, "a") as f:
          f.write(metrics_row)
    print(k)
    print(sse_cost[k])
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,10),sse_cost[2:10])
ax.set_xlabel('k')
ax.set_ylabel('cost')
# paveikslis , pagalvojau gal 6 klasteriai?
fig.savefig('Cost.png')
# parinaku 6 klasteriius ir klassterizuoju
k = 6
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(featurized_clustering_df)
clustered_kmeans_df = model.transform(featurized_clustering_df)
centers = model.clusterCenters()
scaler_mean = model_scaler.mean
scaler_std = model_scaler.std
cluster_sizes = model.summary.clusterSizes
n_obs = clustering_df.count()
denormalized_cluster_centers = [
    (cluster_id,) + (size, 100 * size / n_obs) + tuple(center * scaler_std + scaler_mean)
    for cluster_id, (size, center) in 
    enumerate(zip(cluster_sizes, centers))
]
# centus saugome faile
cluster_centers_pddf = pd.DataFrame.from_records(denormalized_cluster_centers)
cluster_centers_pddf.columns = (
    ["cluster_id", "cluster_size", "cluster_size_pct"] + 
    columns_clustering_features
)
path_cluster_centers = "../data/cluster_centers_kmeans__k_{}.csv".format(k)
cluster_centers_pddf.to_csv(path_cluster_centers, index=False)

path_clustered_df = "../data/clustered_kmeans__k_{}_parquet".format(k)
# modeli saugome parquet faile
clustered_kmeans_df.write.parquet(path_clustered_df)


print("stop")
spark.stop()
