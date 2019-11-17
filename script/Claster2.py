import pyspark
import matplotlib.pyplot as plt
from pyspark.sql import functions
import os
import sys
import findspark
import time
import json
import os
import numpy as np
import operator
import jsonlines
import pandas as pd
import pyspark
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.clustering import KMeans, GaussianMixture
import pyspark.ml.feature as feat
from IPython.display import Image
import plotnine as gg
import pandas as pd
import json
from IPython.display import Image
import plotnine as gg
import pandas as pd
import json
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
clustering_df = spark.read.parquet(path_aggregated_df)
columns_clustering_features = [
"calls_outgoing_count",
"user_spendings",
"sms_incoming_count",
"user_use_gprs",
"sms_outgoing_count",
"user_no_outgoing_activity_in_days",
"calls_outgoing_spendings",
"user_lifetime"
]
 
print("before assemble")
# duomenu paruosimas
vector_assembler = VectorAssembler(
    inputCols=columns_clustering_features, 
    outputCol="initial_features")


standard_scaler = StandardScaler(
    inputCol="initial_features", 
    outputCol="features", 
    withStd=True, 
    withMean=True)
print("after scale")
vectorized_df = vector_assembler.transform(clustering_df)
model_scaler = standard_scaler.fit(vectorized_df)
featurized_clustering_df = model_scaler.transform(vectorized_df)
featurization_pipeline = Pipeline(stages=[vector_assembler, standard_scaler])
featurization_pipeline_model = featurization_pipeline.fit(clustering_df)
model_scaler = featurization_pipeline_model.stages[-1]
featurized_clustering_df = featurization_pipeline_model.transform(clustering_df)
sse_cost = np.zeros(20)
k = 5
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