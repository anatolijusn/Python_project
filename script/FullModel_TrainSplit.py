import pyspark
from pyspark.sql import Row
from pyspark.sql.functions import col
import pyspark
from pyspark.ml import classification
from pyspark.ml import feature
from pyspark.ml import evaluation
from pyspark.ml import pipeline
import itertools
import json
import os
import sys
import pathlib
import uuid

import pyspark
from pyspark.ml import classification
from pyspark.ml import feature
from pyspark.ml import evaluation
from pyspark.ml import pipeline
spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("Python Spark simple decicion tree model")
    .enableHiveSupport()
    .getOrCreate()
)

rt_path="C:/Users/37065/Source/Repos/PythonApplication4"
temp=pathlib.Path(rt_path)

#print(os.path.join(path, "/home", "file.txt"))
print("start")
temp1=rt_path+"/data/clustered_kmeans__k_5_parquet/"
temp2=rt_path+"/data/sample_aggregated_usage_with_churn/"
data_clustered = spark.read.parquet(temp1)
data_df = spark.read.parquet(temp2)
data_df.columns
print("end")
path_training =rt_path+"/data/sample_aggregated_usage_with_churn/training_full_parquet"
path_validation = rt_path+"/data/sample_aggregated_usage_with_churn/validation_full_parquet"
path_test =rt_path+"/data/sample_aggregated_usage_with_churn/test_full_parquet"
data_df.createOrReplaceTempView("t2")
temp_df=spark.sql("""
                      SELECT user_account_id,churn,calls_outgoing_spendings,calls_outgoing_count,user_spendings,sms_incoming_count,calls_outgoing_duration_max,
                             calls_outgoing_spendings_max,user_account_balance_last,
                             sms_outgoing_to_onnet_count,reloads_inactive_days,reloads_count,
                             sms_outgoing_count,sms_outgoing_spendings_max,sms_outgoing_to_abroad_count,sms_outgoing_spendings,
                             gprs_spendings,sms_incoming_spendings,calls_outgoing_to_onnet_count,
                             gprs_session_count,user_no_outgoing_activity_in_days,user_has_outgoing_sms,user_has_outgoing_calls,
                             user_intake,user_use_gprs,user_does_reload
               
                      FROM t2
                       """)
training_df, validation_df, test_df = temp_df.randomSplit([0.7, 0.15, 0.15], seed=10000)
print(path_training)
training_df.write.parquet(path_training)
validation_df.write.parquet(path_validation)
test_df.write.parquet(path_test)
spark.stop()