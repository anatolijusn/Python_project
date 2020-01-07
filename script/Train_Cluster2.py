
import pyspark
from pyspark.sql import Row
from pyspark.sql.functions import col
import pyspark
from pyspark.ml import classification
from pyspark.ml import feature
from pyspark.ml import evaluation
from pyspark.ml import pipeline
import os

import pyspark
from pyspark.sql import Row
from pyspark.sql.functions import col
import pyspark
from pyspark.ml import classification
from pyspark.ml import feature
import pyspark
from pyspark.sql import Row
from pyspark.sql.functions import col
import pyspark
from pyspark.ml import classification
from pyspark.ml import feature
from pyspark.ml import evaluation
from pyspark.ml import pipeline
import pathlib
from pathlib import Path
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
path_training =rt_path+"/data/sample_aggregated_usage_with_churn/training_{}_parquet"
path_validation = rt_path+"/data/sample_aggregated_usage_with_churn/validation_{}_parquet"
path_test =rt_path+"/data/sample_aggregated_usage_with_churn/test_{}_parquet"
data_df.createOrReplaceTempView("t2")

#dir_models = "/home/vagrant/synced_dir/ktu-p160m132-fall2019-spark-examples/models/classification_full_data"
for i in range(0,5):
    if i==2:
        continue
    data_clistered_cluster=data_clustered[data_clustered["prediction"]==i].select("user_account_id")
    data_clistered_cluster.count()
    data_clistered_cluster.createOrReplaceTempView("t1")
   
    temp_df=spark.sql("""
                      SELECT churn,calls_outgoing_spendings,calls_outgoing_count,user_spendings,sms_incoming_count,calls_outgoing_duration_max,
                             calls_outgoing_spendings_max,user_account_balance_last,
                             sms_outgoing_to_onnet_count,reloads_inactive_days,reloads_count,
                             sms_outgoing_count,sms_outgoing_spendings_max,sms_outgoing_to_abroad_count,sms_outgoing_spendings,
                             gprs_spendings,sms_incoming_spendings,calls_outgoing_to_onnet_count,
                             gprs_session_count,user_no_outgoing_activity_in_days,user_has_outgoing_sms,user_has_outgoing_calls,
                             user_intake,user_use_gprs,user_does_reload
               
                      FROM t2
                      INNER JOIN t1 ON t1.user_account_id = t2.user_account_id
                       """)
    #temp_df=temp_df.drop('user_account_id')
    #temp_df=temp_df.drop('user_lifetime')
    print(temp_df.columns)
    training_df, validation_df, test_df = temp_df.randomSplit(
    [0.7, 0.15, 0.15], 
    seed=10000)
    print(path_training.format(i))
    training_df.write.parquet(path_training.format(i))
    validation_df.write.parquet(path_validation.format(i))
    test_df.write.parquet(path_test.format(i))
spark.stop()