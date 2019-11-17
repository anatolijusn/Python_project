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
def compute_fk(k, sse, prev_sse, dim):
    if k == 1 or prev_sse == 0:
        return 1
    weight = weight_factor(k, dim)
    return sse / (weight * prev_sse)

# calculating alpha_k in functional style with tail recursion -- which is not optimized in Python :(
def weight_factor(k, dim):
    if not k > 1:
        raise ValueError("k must be greater than 1")
        
    def weigth_factor_accumulator(acc, k):
        if k == 2:
            return acc
        return weigth_factor_accumulator(acc + (1 - acc) / 6, k - 1)
        
    weight_k2 = 1 - 3 / (4 * dim)
    return weigth_factor_accumulator(weight_k2, k)
def compute_fk_from_k_sse_pairs(k_sse_pairs, dimension):
    triples = make_fk_triples(k_sse_pairs)
    k_fk_pairs = [
        (k, compute_fk(k, sse, prev_sse, dimension))
        for (k, sse, prev_sse) in triples]
    return sorted(k_fk_pairs, key=lambda pair: pair[0])


def make_fk_triples(k_sse_pairs):
    sorted_pairs = sorted(k_sse_pairs, reverse=True)
    candidates = list(zip(sorted_pairs, sorted_pairs[1:] + [(0, 0.0)]))
    triples = [
        (k, sse, prev_sse)
        for ((k, sse), (prev_k, prev_sse)) in candidates
        if k - prev_k == 1
    ]
    return triples



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
#



#


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


path_metrics_kmeans_sse=""
# cia keliai skaitomi is json failo, jau dirbame su parquet failu
json_file_path = "Params.json"
with open(json_file_path, 'r') as j:
     contents = json.load(j)
cluster=contents['cluster']
for item in cluster:
    path_metrics_kmeans_sse=item['path_metrics_kmeans_sse']
metrics_pddf = pd.read_json(
     path_metrics_kmeans_sse, 
    orient="records",
    lines=True)
k_sse_pddf = metrics_pddf[["k", "sse"]]
dimension = 8
k_sse_pairs = [tuple(r) for r in k_sse_pddf.to_records(index=False)]
k_fk_pairs = compute_fk_from_k_sse_pairs(k_sse_pairs, dimension)
k_fk_pddf = pd.DataFrame.from_records(k_fk_pairs, columns=["k", "fk"])

plot_k_sse = (
    gg.ggplot(gg.aes(x="k", y="sse"), data=k_sse_pddf) + 
    gg.geom_line() + 
    gg.xlab("K") +
    gg.ylab("SSE") + 
    gg.ggtitle("SSE pagal klasterių skaičių K") +
    gg.theme_bw()
)

plot_k_fk = (
    gg.ggplot(gg.aes(x="k", y="fk"), data=k_fk_pddf) + 
    gg.geom_line() + 
    gg.xlab("K") +
    gg.ylab("f(K)") + 
    gg.ggtitle("f(K) pagal klasterių skaičių K") +
    gg.theme_bw()
)
print(plot_k_sse)
print(plot_k_fk)
plot_k_sse.save("k_sse.png")
plot_k_fk.save("k_fk.png")


print("stop")
spark.stop()
