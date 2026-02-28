# park_session.py
# Spark functions to increase CPU core and RAM usage for performance on high computation tasks

from pyspark.sql import SparkSession

def get_spark(app_name: str = "recsys") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # use all CPU cores
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    return spark

