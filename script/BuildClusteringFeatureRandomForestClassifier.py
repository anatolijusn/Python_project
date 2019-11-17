from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, VectorSlicer
from pyspark.ml.classification import  RandomForestClassifier
import pyspark
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import functions
import json
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import os
import sys
import findspark
import time
import operator
import functools
import plotnine as gg
import pandas as pd
import pyspark.sql.functions as sql_fn
from pyspark.ml import stat
from pyspark.ml import feature
import IPython

import uuid

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
  
#path_aggregated_df = "../data/sample_aggregated_usage_with_churn"
clustering_df = spark.read.parquet(path_aggregated_df)
# destytojas sake kad reikai ismesti dali stulpeliu, na teorishkai reikia nuspresti kurie reiksmingi,
# tame ismesti, turbut butu gerai imesti i json'a, kad galima butu koreguoti nekeiciant kodo
with open('remains_feature.json', 'r') as j:
     columns_clustering_features = json.load(j)

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


rf = RandomForestClassifier(labelCol="churn", featuresCol="initial_features", seed = 8464,
                            numTrees=10, cacheNodeIds = True, subsamplingRate = 0.7)
label_indexes = StringIndexer(inputCol = 'churn', outputCol = 'label', handleInvalid = 'keep')
pipe = Pipeline(stages = [vector_assembler, label_indexes, rf])
mod = pipe.fit(clustering_df)

mod.stages[-1].featureImportances
df2 = mod.transform(clustering_df)
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
features=ExtractFeatureImp(mod.stages[-1].featureImportances, df2, "initial_features").head(30)
with open('usefulfeature.json', 'w') as f:
    json.dump(features.to_json(), f)

