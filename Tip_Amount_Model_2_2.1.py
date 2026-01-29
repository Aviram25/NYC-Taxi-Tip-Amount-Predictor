#!/usr/bin/env python3
"""
Tip Range Prediction Model (Classification)

This script implements a classification model to predict the range of the tip amount.

Classes (Target Bins):
    * 0 (No Tip): tip = 0
    * 1 (Low): 0 < tip <= 3
    * 2 (Medium): 3 < tip <= 6
    * 3 (High): 6 < tip <= 10
    * 4 (Very High): tip > 10

Models:
    1. Random Forest Classifier
    2. XGBoost Classifier
    3. Random Forest Classifier with Hyperparameter Tuning
"""

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, Bucketizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from xgboost.spark import SparkXGBClassifier


def create_spark_session():
    """Create and configure Spark session"""
    spark = SparkSession.builder \
        .appName("Taxi Tip Classification") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "3g") \
        .config("spark.executor.instances", "3") \
        .config("spark.executor.memory", "3g") \
        .getOrCreate()
    return spark


def load_data(spark, data_path="clean_test"):
    """Load data from parquet file"""
    df = spark.read.parquet(data_path)
    print(f"Total records: {df.count()}")
    return df


def feature_engineering(df):
    """Perform feature engineering and target binning"""
    
    # Drop missing values
    df = df.dropna(subset=[
        "fare_amount", "trip_distance", "payment_type",
        "tip_amount", "tpep_pickup_datetime", "tpep_dropoff_datetime"
    ])
    
    # 1. Time Features
    df = df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime")) \
           .withColumn("pickup_day", F.dayofweek("tpep_pickup_datetime")) \
           .withColumn("pickup_month", F.month("tpep_pickup_datetime")) \
           .withColumn("is_weekend", F.when(F.col("pickup_day").isin([1, 7]), 1).otherwise(0))
    
    # 2. Trip Duration
    df = df.withColumn(
        "trip_duration",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60
    ).filter((F.col("trip_duration") > 0) & (F.col("trip_duration") < 300))
    
    # 3. Target Binning (Tip Class)
    # 0: No Tip (0)
    # 1: Low (0 - 3]
    # 2: Medium (3 - 6]
    # 3: High (6 - 10]
    # 4: Very High (> 10)
    
    bucketizer = Bucketizer(
        splits=[-float("inf"), 0.0001, 3, 6, 10, float("inf")],
        inputCol="tip_amount",
        outputCol="tip_class_raw"
    )
    
    df = bucketizer.transform(df)
    df = df.withColumn("label", F.col("tip_class_raw").cast("double"))
    
    # Check class distribution
    print("\nClass Distribution:")
    df.groupBy("label").count().orderBy("label").show()
    
    return df


def prepare_features(df):
    """Prepare feature vectors for ML"""
    
    # Index payment type
    indexer = StringIndexer(inputCol="payment_type", outputCol="payment_type_idx", handleInvalid="keep")
    df = indexer.fit(df).transform(df)
    
    # Define feature columns
    feature_cols = [
        "fare_amount", "trip_distance", "trip_duration",
        "pickup_hour", "pickup_day", "pickup_month",
        "is_weekend", "payment_type_idx"
    ]
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_ml = assembler.transform(df).select("features", "label")
    
    # Split data
    train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
    
    return train_df, test_df


def train_random_forest(train_df, test_df):
    """Train Random Forest Classifier"""
    
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)
    
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=20,
        maxDepth=5,
        seed=42
    )
    
    rf_model = rf.fit(train_df)
    rf_preds = rf_model.transform(test_df)
    
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    
    accuracy = evaluator_acc.evaluate(rf_preds)
    f1_score = evaluator_f1.evaluate(rf_preds)
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print(f"Random Forest F1 Score: {f1_score:.4f}")
    
    return rf_model, accuracy, f1_score


def train_xgboost(train_df, test_df):
    """Train XGBoost Classifier"""
    
    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)
    
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        num_workers=2,
        use_gpu=False
    )
    
    xgb_model = xgb.fit(train_df)
    xgb_preds = xgb_model.transform(test_df)
    
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    
    accuracy = evaluator_acc.evaluate(xgb_preds)
    f1_score = evaluator_f1.evaluate(xgb_preds)
    
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print(f"XGBoost F1 Score: {f1_score:.4f}")
    
    return xgb_model, accuracy, f1_score


def train_tuned_random_forest(train_df, test_df):
    """Train Random Forest with Hyperparameter Tuning"""
    
    print("\n" + "="*60)
    print("Training Random Forest with Hyperparameter Tuning")
    print("="*60)
    
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        seed=42
    )
    
    paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100])
        .addGrid(rf.maxDepth, [5, 10])
        .build()
    )
    
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator_acc,
        numFolds=3,
        parallelism=2
    )
    
    # Cache training data for CV
    train_df.cache()
    train_df.count()
    
    print("Running cross-validation (this may take a while)...")
    cv_model = cv.fit(train_df)
    
    rf_preds = cv_model.bestModel.transform(test_df)
    
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    
    accuracy = evaluator_acc.evaluate(rf_preds)
    f1_score = evaluator_f1.evaluate(rf_preds)
    
    print(f"Best RF Accuracy: {accuracy:.4f}")
    print(f"Best RF F1 Score: {f1_score:.4f}")
    
    # Print best parameters
    best_model = cv_model.bestModel
    print(f"\nBest Parameters:")
    print(f"  numTrees: {best_model.getNumTrees}")
    print(f"  maxDepth: {best_model.getMaxDepth()}")
    
    return cv_model, accuracy, f1_score


def main():
    """Main execution function"""
    
    print("="*60)
    print("NYC Taxi Tip Range Prediction Model")
    print("="*60)
    
    # Create Spark session
    spark = create_spark_session()
    
    # Load data
    data_path = "clean_test"
    df = load_data(spark, data_path)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prepare features
    train_df, test_df = prepare_features(df)
    
    # Train models
    results = {}
    
    # 1. Random Forest
    rf_model, rf_acc, rf_f1 = train_random_forest(train_df, test_df)
    results['Random Forest'] = {'accuracy': rf_acc, 'f1_score': rf_f1}
    
    # 2. XGBoost
    xgb_model, xgb_acc, xgb_f1 = train_xgboost(train_df, test_df)
    results['XGBoost'] = {'accuracy': xgb_acc, 'f1_score': xgb_f1}
    
    # 3. Tuned Random Forest
    tuned_rf_model, tuned_rf_acc, tuned_rf_f1 = train_tuned_random_forest(train_df, test_df)
    results['Tuned RF'] = {'accuracy': tuned_rf_acc, 'f1_score': tuned_rf_f1}
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<15} {'F1 Score':<15}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<15.4f} {metrics['f1_score']:<15.4f}")
    print("="*60)
    
    # Note
    print("\nNote: SparkXGBClassifier cannot be used with ParamGridBuilder or CrossValidator.")
    print("That's why we are using Random Forest Classifier only for hyperparameter tuning.")
    
    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main()
