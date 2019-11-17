import os
import sys
import findspark
import time
import json
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import operator
import functools
import plotnine as gg
import pyspark
import pandas as pd
import pyspark.sql.functions as sql_fn
from pyspark.ml import stat
from pyspark.ml import feature
import IPython
import numpy as np
import uuid

def compute_corr(df, columns, method="pearson"):
    assembler = feature.VectorAssembler(inputCols=columns, 
        outputCol="featuresCorrelation")
    corr_featurized_df = assembler.transform(df)
    corr_df = stat.Correlation.corr(corr_featurized_df, "featuresCorrelation", method)
    corr_matrix = corr_df.first()[0].toArray()
    corr_pddf = pd.DataFrame(corr_matrix, columns=columns, index=columns)
    return corr_pddf
findspark.init()

spark = (SparkSession
    .builder
    .appName("ClusterPrj")
    .enableHiveSupport()
    .getOrCreate())
print("start visualization")
customer_usage = ""
customer_churn = ""
#parametrus skaitau is json failo cia jau po "sampling" gauti failai
json_file_path = "Params.json"
with open(json_file_path, 'r') as j:
     contents = json.load(j)

customer_usage = contents['customer_usage']
customer_churn = contents['customer_churn']
print(customer_usage)
print(customer_churn)
usage_df = spark.read.csv(customer_usage,header=True,inferSchema=True)
summary_usage_df = usage_df.describe()
# bandau gauti descritpion statistikos aprasyma ir irasyti ji csv

(summary_usage_df
    .toPandas()
    .transpose()
    .to_csv("summary__usage.csv", 
            index=True, header=False))
pd.options.display.max_columns = 200
#sio metu turime tik 2 stulpelius histogramai aprasyti. Turbut mazai. Reiketu dar prideti?
histogram_columns =  [
    "calls_outgoing_to_onnet_spendings", 
    "sms_outgoing_spendings",
    'user_spendings',
    'sms_outgoing_to_abroad_spendings',
    'calls_outgoing_to_onnet_spendings',
    'gprs_spendings'
]
columns_correlation = [
    c for c in usage_df.columns 
    if c not in {"user_account_id", "year"}
]

# irasau i csv faila koreliacijos rezultatus
corr_pearson_pddf = compute_corr(usage_df, columns_correlation)
corr_spearman_pddf = compute_corr(usage_df, columns_correlation, "spearman")
corr_pearson_pddf.to_csv("corr_pearson__usage.csv", index=True, header=True)
corr_spearman_pddf.to_csv("corr_spearman__usage.csv", index=True, header=True)
corr_matrix=corr_pearson_pddf.abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
remains_feature_list=[x for x in usage_df.columns if x not in to_drop]
with open('remains_feature.json', 'w') as f:
    json.dump(remains_feature_list, f)
histogram_columns =  remains_feature_list
n_bins = 30
histogram_dataset = [
    (c, usage_df.select(c).rdd.map(lambda r: r[0]).histogram(n_bins),) 
    for c in histogram_columns
]

histogram_pddfs = [
    (column, pd.DataFrame.from_records(zip(xs, ys + [0]), columns=["x", "y"]))
    for column, (xs, ys) in histogram_dataset
]

# rasau histograma ir piesini i kataloga
for column, hist_df in histogram_pddfs:
    hist_df.to_csv("hist__{}.csv".format(column), index=False)
    plot_hist = (gg.ggplot(gg.aes(x="x", weight="y"), hist_df) + gg.geom_bar() + gg.xlab(column) + gg.ylab("dažnis") + gg.ggtitle("'{}' histograma".format(column)) + gg.theme_bw())

    plot_hist.save("hist__{}.png".format(column))




print("stop")
spark.stop()