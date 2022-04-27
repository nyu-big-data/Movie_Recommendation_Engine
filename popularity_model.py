from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F
from pyspark.sql.functions import lit, col, mean, count, udf

def weighted_rating(v,r, per, avg):
    return float((v/(v+per) * r) + (per/(per+v) * avg))

def main(spark):

    ratings = spark.read.option("header", False).csv('hdfs:/user/hb2474/ml-latest-small/training_data_small.csv')
    ratings = ratings.toDF('userId', 'movieId', 'rating', 'unknown', 'split')
    ratings = ratings.withColumn('rating', ratings['rating'].cast(DoubleType()))
    agg_ratings = ratings.groupBy('movieId').agg(mean('rating').alias('avg_rating'), count('userId').alias('user_count')).select('movieId', 'avg_rating', 'user_count')
    m = agg_ratings.select(mean('avg_rating')).collect()[0][0]
    quant = agg_ratings.select(F.expr('percentile_approx(user_count, 0.9)')).collect()[0][0]
    agg_ratings = agg_ratings.filter(agg_ratings.user_count >= quant)

    weighted_score = udf(weighted_rating, DoubleType())
    agg_ratings = agg_ratings.withColumn('score', weighted_score('user_count','avg_rating',  lit(quant), lit(m))).select('movieId','avg_rating', 'user_count', 'score')
    agg_ratings = agg_ratings.orderBy(col("score").desc()).select('movieId')
    agg_ratings.show()


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)