import os
import sys
import findspark
import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import uuid
findspark.init()

spark = (
     SparkSession
    .builder
    .appName("ClusterPrj")
    .enableHiveSupport()
    .getOrCreate()
)

usage_df = spark.read.csv("C:\SymCache\Data\customer_usage_00003.csv", header=True)
cn=usage_df.count()
ln=len(usage_df.columns)
print(usage_df.columns)
#sitie keliai turetu atsidurti json faile.Siuo metu nezinome varianto, tai kai jis atsiras 

churn_df = spark.read.csv("C:\SymCache\Data\customer_churn_00003.csv", header=True)
chr_cnt=churn_df.count()
print(churn_df.columns)
#Bandymas sugeneruoti faila
churn_sample_df = churn_df.sample(False, 0.1, seed=111111)
print(churn_sample_df.count())
churn_sample_df.createOrReplaceTempView("churn_sample")
usage_df.createOrReplaceTempView("usage_full")
query = """
SELECT usage_full.*
FROM churn_sample
JOIN usage_full ON usage_full.user_account_id = churn_sample.user_account_id
"""
usage_churn_sample_df = spark.sql(query)
print(usage_churn_sample_df.columns == usage_df.columns)
print(usage_churn_sample_df.count())
sample_uuid = uuid.uuid4()
dir_output="C:/SymCache/Data/"
path_tmp_output_usage = os.path.join(
        dir_output,
        "tmp_usage_sample_{}".format(sample_uuid))
path_tmp_output_churn = os.path.join(
        dir_output,
        "tmp_churn_sample_{}".format(sample_uuid))
# kadangi gavosi kataloge 4 usage failai tai noredamas issaugot i viena panaudojau tokia funkcija
usage_churn_sample_df.coalesce(1).write.csv(path_tmp_output_usage,header='true')
churn_sample_df.coalesce(1).write.csv(path_tmp_output_churn,header='true')
spark.stop()
