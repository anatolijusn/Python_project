import pyspark

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
    .appName("Python Spark SQL aggregation with join")
    .enableHiveSupport()
    .getOrCreate()
)
# sitame faile skaitome 2 paruostus failus po "sampling", keliai nurodyti json faile,  tada apdorojame stulpelius
# po duomenu grupavimo idedam, issaugom viska parquer formate 

print("start spark")
customer_usage = ""
customer_churn = ""

json_file_path = "Params.json"
with open(json_file_path, 'r') as j:
     contents = json.load(j)

customer_usage = contents['customer_usage']
customer_churn = contents['customer_churn']
usage_df = spark.read.csv(
    customer_usage, 
    header=True, 
    inferSchema=True)

churn_df = spark.read.csv(
    customer_churn, 
    header=True, 
    inferSchema=True)

all_usage_columns = usage_df.columns

date_columns = ["year", "month"]
id_columns = ["user_account_id"]
binary_columns = [
    "user_intake",
    "user_has_outgoing_calls", 
    "user_has_outgoing_sms", 
    "user_use_gprs", 
    "user_does_reload"
]
categorical_columns = date_columns + binary_columns + id_columns
continuous_columns = [c for c in all_usage_columns if c not in categorical_columns]
expressions_avg = [functions.avg(usage_df[c]).alias(c) for c in continuous_columns]
sql_expressions_max = [functions.max(usage_df[c]).alias(c) for c in binary_columns]
sql_expressions_count = [functions.count("*").alias("n_months")]
print("before aggregation")
expressions_aggregation = expressions_avg + sql_expressions_max + sql_expressions_count
agg_usage_churn_df = (
    usage_df
    .groupBy(usage_df["user_account_id"])
    .agg(*expressions_aggregation)
    .join(churn_df, "user_account_id")
)
print("after aggreg")
path_aggregated_joined_usage_churn_df = "../data/sample_aggregated_usage_with_churn"
agg_usage_churn_df.write.parquet(path_aggregated_joined_usage_churn_df)

spark.stop()