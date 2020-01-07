
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
i=0;
rt_path="C:/Users/37065/Source/Repos/PythonApplication4"
path_training = rt_path+"/data/sample_aggregated_usage_with_churn/training_full_parquet"
path_validation = rt_path+"/data/sample_aggregated_usage_with_churn/validation_full_parquet"
path_test =rt_path + "/data/sample_aggregated_usage_with_churn/test_full_parquet"
dir_models =rt_path + "/data/models/classification_all_data"
clustered_dir_models=rt_path + "/data/models/classification_clusterd_all_data"
training_df=  spark.read.parquet(path_training)
validation_df=spark.read.parquet(path_validation)
test_df=spark.read.parquet(path_test)

#---------------------------------------------------

cls_path_test_0=rt_path+"\\data\\models\\predict\\test_clustering_full_data_0_parquet\\5"
cls_path_validation_0=rt_path+"\\data\\models\\predict\\validation_clustering_full_data_0_parquet\\5"

cls_path_test_0_df=spark.read.parquet(cls_path_test_0)
cls_path_validation_0_df=spark.read.parquet(cls_path_validation_0)


cls_path_test_1=rt_path+"\\data\\models\\predict\\test_clustering_full_data_1_parquet\\5"
cls_path_validation_1=rt_path+"\\data\\models\\predict\\validation_clustering_full_data_1_parquet\\5"


cls_path_test_1_df=spark.read.parquet(cls_path_test_1)
cls_path_validation_1_df=spark.read.parquet(cls_path_validation_1)


cls_path_test_3=rt_path+"\\data\\models\\predict\\test_clustering_full_data_3_parquet\\5"
cls_path_validation_3=rt_path+"\\data\\models\\predict\\validation_clustering_full_data_3_parquet\\5"


cls_path_test_3_df=spark.read.parquet(cls_path_test_3)
cls_path_validation_3_df=spark.read.parquet(cls_path_validation_3)



cls_path_test_4=rt_path+"\\data\\models\\predict\\test_clustering_full_data_4_parquet\\5"
cls_path_validation_4=rt_path+"\\data\\models\\predict\\validation_clustering_full_data_4_parquet\\5"




cls_path_test_4_df=spark.read.parquet(cls_path_test_4)
cls_path_validation_4_df=spark.read.parquet(cls_path_validation_4)



cls_result_validation_df=cls_path_validation_0_df.union(cls_path_validation_1_df)
cls_result_validation_df=cls_result_validation_df.union(cls_path_validation_3_df)
cls_result_validation_df=cls_result_validation_df.union(cls_path_validation_4_df)


cls_result_test_df=cls_path_test_0_df.union(cls_path_test_1_df)
cls_result_test_df=cls_result_test_df.union(cls_path_test_3_df)
cls_result_test_df=cls_result_test_df.union(cls_path_test_4_df)


accuracy_evaluator = evaluation.MulticlassClassificationEvaluator(
    metricName="accuracy", labelCol="churn", predictionCol="prediction")

precision_evaluator = evaluation.MulticlassClassificationEvaluator(
                                  metricName="weightedPrecision", labelCol="churn")
recall_evaluator = evaluation.MulticlassClassificationEvaluator(
                               metricName="weightedRecall", labelCol="churn",
                               predictionCol="prediction")
auroc_evaluator = evaluation.BinaryClassificationEvaluator(metricName='areaUnderROC', 
                                                                       labelCol="churn")
f1_evaluator = evaluation.MulticlassClassificationEvaluator(
    metricName="f1", labelCol="churn", predictionCol="prediction")

cls_test_metrics = {
    "accuracy": accuracy_evaluator.evaluate(cls_result_test_df),
    "precision": precision_evaluator.evaluate(cls_result_test_df),
    "recall": recall_evaluator.evaluate(cls_result_test_df),
    "f1": f1_evaluator.evaluate(cls_result_test_df),
    "auroc": auroc_evaluator.evaluate(cls_result_test_df)
     }

cls_validation_metrics = {
    "accuracy": accuracy_evaluator.evaluate(cls_result_validation_df),
    "precision": precision_evaluator.evaluate(cls_result_validation_df),
    "recall": recall_evaluator.evaluate(cls_result_validation_df),
    "f1": f1_evaluator.evaluate(cls_result_validation_df),
    "auroc": auroc_evaluator.evaluate(cls_result_validation_df)
     }

cls_model_id = uuid.uuid4()
clustered_dir_models = pathlib.Path(clustered_dir_models).joinpath(str(cls_model_id))
clustered_dir_models.mkdir(parents=True, exist_ok=True)
path_cls_test_metrics=pathlib.Path(clustered_dir_models).joinpath("metrics_test.json")
path_cls_validation_metrics = pathlib.Path(clustered_dir_models).joinpath("metrics_validation.json")
with open(path_cls_validation_metrics, "w") as f:
         json.dump(cls_validation_metrics, f)
with open(path_cls_test_metrics, "w") as f:
         json.dump(cls_test_metrics, f)
    
    
#---------------------------------------------------





params = {
    "path_input": "/home/vagrant/synced_dir/churn-prediction/data/interim/aggregated_raw",
    "dir_models": "/home/vagrant/synced_dir/churn-prediction/models/classification__001",
    "hiperparameter_grid": {
        "numTrees": [10, 30, 100],
        "maxDepth": [5, 10],
    },
    "feature_columns": [
       # 'user_lifetime',
        'user_no_outgoing_activity_in_days',
        'user_account_balance_last',
        'user_spendings',
        'reloads_inactive_days',
        'reloads_count',
        'reloads_sum',
        'calls_outgoing_count',
        'calls_outgoing_spendings',
        'calls_outgoing_duration',
        'calls_outgoing_spendings_max',
        'calls_outgoing_duration_max',
        'calls_outgoing_inactive_days',
        'calls_outgoing_to_onnet_count',
        'calls_outgoing_to_onnet_spendings',
        'calls_outgoing_to_onnet_duration',
        'calls_outgoing_to_onnet_inactive_days',
        'calls_outgoing_to_offnet_count',
        'calls_outgoing_to_offnet_spendings',
        'calls_outgoing_to_offnet_duration',
        'calls_outgoing_to_offnet_inactive_days',
        'calls_outgoing_to_abroad_count',
        'calls_outgoing_to_abroad_spendings',
        'calls_outgoing_to_abroad_duration',
        'calls_outgoing_to_abroad_inactive_days',
        'sms_outgoing_count',
        'sms_outgoing_spendings',
        'sms_outgoing_spendings_max',
        'sms_outgoing_inactive_days',
        'sms_outgoing_to_onnet_count',
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
        'last_100_calls_outgoing_duration',
        'last_100_calls_outgoing_to_onnet_duration',
        'last_100_calls_outgoing_to_offnet_duration',
        'last_100_calls_outgoing_to_abroad_duration',
        'last_100_sms_outgoing_count',
        'last_100_sms_outgoing_to_onnet_count',
        'last_100_sms_outgoing_to_offnet_count',
        'last_100_sms_outgoing_to_abroad_count',
        'last_100_gprs_usage',
        'user_intake',
        'user_has_outgoing_calls',
        'user_has_outgoing_sms',
        'user_use_gprs',
        'user_does_reload',
        'n_months',
    ]
}


def make_param_sets(grid):
    pairs = [
        [(param_name, value) for value in param_values]
        for param_name, param_values in grid.items()
    ]
    return [dict(s) for s in itertools.product(*pairs)]

all_columns =validation_df.columns
date_columns = ["year", "month"]
#id_columns = ["user_account_id"]
binary_columns = [
    "user_intake",
    "user_has_outgoing_calls", 
    "user_has_outgoing_sms", 
    "user_use_gprs", 
    "user_does_reload"
]


response_columns = ["churn"]
#categorical_columns = date_columns + binary_columns #+ id_columns
categorical_columns =binary_columns #+ id_columns
continuous_columns = [
    c for c in all_columns
    if c not in categorical_columns + response_columns
]
feature_columns = binary_columns + categorical_columns + continuous_columns
hiperparam_sets = make_param_sets(params["hiperparameter_grid"])
assembler = feature.VectorAssembler(inputCols=feature_columns, outputCol="features")
for hiperparams in hiperparam_sets:
    
    rf_params=hiperparams
    classific=classification.RandomForestClassifier(
    labelCol="churn", **rf_params)
    dt_pipe_md = pipeline.Pipeline(stages=[
                                                     assembler, 
                                                     classific ])
    dt_pipe_md_model = dt_pipe_md.fit(training_df)
    train_predictions_df = dt_pipe_md_model.transform(training_df)
    validation_predictions_df  = dt_pipe_md_model.transform(validation_df)
    test_prediction_df =dt_pipe_md_model.transform(test_df)
    
    
    accuracy_evaluator = evaluation.MulticlassClassificationEvaluator(
    metricName="accuracy", labelCol="churn", predictionCol="prediction")

    
    precision_evaluator = evaluation.MulticlassClassificationEvaluator(
                                  metricName="weightedPrecision", labelCol="churn")
    recall_evaluator = evaluation.MulticlassClassificationEvaluator(
                               metricName="weightedRecall", labelCol="churn",
                               predictionCol="prediction")
    auroc_evaluator = evaluation.BinaryClassificationEvaluator(metricName='areaUnderROC', 
                                                                       labelCol="churn")
    
    f1_evaluator = evaluation.MulticlassClassificationEvaluator(
    metricName="f1", labelCol="churn", predictionCol="prediction")
    
    train_metrics = {
    "accuracy": accuracy_evaluator.evaluate(train_predictions_df),
    "precision": precision_evaluator.evaluate(train_predictions_df),
    "recall": recall_evaluator.evaluate(train_predictions_df),
    "f1": f1_evaluator.evaluate(train_predictions_df),
    "auroc": auroc_evaluator.evaluate(train_predictions_df)
     }


    test_metrics = {
    "accuracy": accuracy_evaluator.evaluate(test_prediction_df),
    "precision": precision_evaluator.evaluate(test_prediction_df),
    "recall": recall_evaluator.evaluate(test_prediction_df),
    "f1": f1_evaluator.evaluate(test_prediction_df),
    "auroc": auroc_evaluator.evaluate(test_prediction_df)
     }


    validation_metrics = {
    "accuracy": accuracy_evaluator.evaluate(validation_predictions_df),
    "precision": precision_evaluator.evaluate(validation_predictions_df),
    "recall": recall_evaluator.evaluate(validation_predictions_df),
    "f1": f1_evaluator.evaluate(validation_predictions_df),
    "auroc": auroc_evaluator.evaluate(validation_predictions_df)
     }




    rf_model = dt_pipe_md_model.stages[-1]
    model_params = rf_model.extractParamMap()
    model_description = {
    "name":  "Random Forest",
    "params": {param.name: value for param, value in model_params.items()},
    }
    model_id = uuid.uuid4()
   # k_{}.csv".format(k)
    dir_model = pathlib.Path(dir_models).joinpath(str(model_id))
    dir_model.mkdir(parents=True, exist_ok=True)
    path_pipeline_model = pathlib.Path(dir_model).joinpath("pipeline_model")
    path_train_metrics = pathlib.Path(dir_model).joinpath("metrics_train.json")
    path_test_metrics=pathlib.Path(dir_model).joinpath("metrics_test.json")
    path_validation_metrics = pathlib.Path(dir_model).joinpath("metrics_validation.json")
    path_model_description = pathlib.Path(dir_model).joinpath("model_description.json")
    dt_pipe_md_model.save(str(path_pipeline_model))
    with open(path_train_metrics, "w") as f:
         json.dump(train_metrics, f)
    with open(path_validation_metrics, "w") as f:
         json.dump(validation_metrics, f)

    with open(path_test_metrics, "w") as f:
         json.dump(test_metrics, f)
    with open(path_model_description, "w") as f:
         json.dump(model_description, f)
    