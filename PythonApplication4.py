import os
import sys
import findspark
import time
import json
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
def delayed(seconds):
    def f(x):
        time.sleep(seconds)
        return x
    return f


json_file_path = "Params.json"
with open(json_file_path, 'r') as j:
     contents = json.load(j)

customer_usage = contents['customer_usage']
customer_churn = contents['customer_churn']
cluster=contents['cluster']
for item in cluster:
    path_aggregated_df=item['path_aggregated_df']
    path_metrics_kmeans_sse=item['path_metrics_kmeans_sse']
findspark.init()

conf = SparkConf().set("spark.ui.showConsoleProgress", "false")
sc = SparkContext(appName="PythonStatusAPIDemo", conf=conf)
import numpy as np
TOTAL = 1000000
rdd = sc.parallelize(range(10), 10).map(delayed(0))
reduced = rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
result=reduced.map(delayed(0)).collect()
print("Number of random points:", 12500)
print("Job results are:", result)
sc.stop()