from dfCoresettree import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType

spark = SparkSession \
          .builder \
          .appName("APP") \
          .getOrCreate()

json_schema = StructType([
    StructField("x", StringType()),
    StructField("y", StringType())
])

if __name__ == "__main__":
    df = spark \
      .read \
      .format("kafka") \
      .option("kafka.bootstrap.servers", "localhost:9092") \
      .option("subscribe", "json") \
      .load()
    
    df = df.selectExpr("CAST(key AS STRING)", "CAST(value as STRING)")
    rddDF = df.select(from_json(col("value"), json_schema) \
                         .alias("parsed")).selectExpr("parsed.*")
    rddDF = rddDF.selectExpr("CAST(x AS FLOAT)", "CAST(y AS FLOAT)")
    pandas_df = rddDF.toPandas()
    Tree = dfCoreSetTree(pandas_df, 200)
    Tree.fit()
    Coreset_spark = spark.createDataFrame(Tree.coreset)

    Coreset_spark.select(to_json(struct("x", "y")).alias("value")) \
      .write \
      .format("kafka") \
      .option("kafka.bootstrap.servers", "localhost:9092") \
      .option("topic", "spark") \
      .save()
