from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F
from pyspark.sql.functions import lit, col, mean, count, udf
from pyspark.mllib.evaluation import RankingMetrics
import numpy as np


def weighted_rating(v,r, per, avg):
    """
    Computes the weighted rating for each movie by factoring the user count (no. of ratings)
    """
    return float((v/(v+per) * r) + (per/(per+v) * avg))

def top_k_accuracy(top_k_recommendations, user_labels):
    count = 0
    for movie_id in user_labels:
        if movie_id in top_k_recommendations:
            count+=1
    
    return count/len(user_labels)




def main(spark):


    ## Read the Training File
    ratings = spark.read.option("header", False).csv('hdfs:/user/hb2474/ml-latest-small/training_data_small.csv')
    ratings = ratings.toDF('userId', 'movieId', 'rating', 'unknown', 'split')
    ratings = ratings.withColumn('rating', ratings['rating'].cast(DoubleType()))

    ## Compute Avg Rating and User Count
    agg_ratings = ratings.groupBy('movieId').agg(mean('rating').alias('avg_rating'), count('userId').alias('user_count')).select('movieId', 'avg_rating', 'user_count')
    m = agg_ratings.select(mean('avg_rating')).collect()[0][0]
    quant = agg_ratings.select(F.expr('percentile_approx(user_count, 0.9)')).collect()[0][0]
    agg_ratings = agg_ratings.filter(agg_ratings.user_count >= quant)

    ## Computed the weighted score for each movie
    weighted_score = udf(weighted_rating, DoubleType())
    agg_ratings = agg_ratings.withColumn('score', weighted_score('user_count','avg_rating',  lit(quant), lit(m))).select('movieId','avg_rating', 'user_count', 'score')
    agg_ratings = agg_ratings.orderBy(col("score").desc()).select('movieId')

    ## Generate Top - 100 Recommedations
    movie_ids = (np.array(agg_ratings.collect())).reshape(-1)
    
    ## Read the Validation File
    val_ratings = spark.read.option("header", False).csv('hdfs:/user/hb2474/validation_data_small.csv')
    val_ratings = val_ratings.toDF('userId', 'movieId', 'rating', 'unknown', 'split')
    val_ratings = val_ratings.withColumn('rating', val_ratings['rating'].cast(DoubleType()))
    val_ratings = val_ratings.select('movieId').collect()
    val_ratings = np.array(val_ratings).reshape(-1)
    val_acc = top_k_accuracy(movie_ids, val_ratings)

    ## Read the Test File
    test_ratings = spark.read.option("header", False).csv('hdfs:/user/hb2474/test_data_small.csv')
    test_ratings = test_ratings.toDF('userId', 'movieId', 'rating', 'unknown', 'split')
    test_ratings = test_ratings.withColumn('rating', test_ratings['rating'].cast(DoubleType()))
    test_ratings = test_ratings.select('movieId').collect()
    test_ratings = np.array(test_ratings).reshape(-1)
    test_acc = top_k_accuracy(movie_ids, test_ratings)
    
    print('Validation Acc:{}'.format(val_acc))
    print('Test Acc:{}'.format(test_acc))


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)