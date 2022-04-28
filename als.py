from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, explode

def main(spark):

    train = spark.read.option("header", False).csv(f'hdfs:/user/el3418/training_data.csv')
    train = train.toDF('userId', 'movieId', 'rating', 'timestamp', 'split')
    train = train.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', train['rating'].cast('float'))

    test = spark.read.option("header", False).csv(f'hdfs:/user/el3418/test_data.csv')
    test = test.toDF('userId', 'movieId', 'rating', 'timestamp', 'split')
    test = test.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', test['rating'].cast('float'))

    als = ALS(
        maxIter=3,
        rank=5,
        userCol="userId", 
        itemCol="movieId",
        ratingCol="rating", 
        nonnegative = True, 
        implicitPrefs = False,
        coldStartStrategy="drop"
    )

    #Fit cross validator to the 'train' dataset
    model = als.fit(train)
    
    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="rating", 
            predictionCol="prediction") 

    # View the predictions
    test_predictions = model.transform(test)
    RMSE = evaluator.evaluate(test_predictions)
    print(RMSE)

    #Output in the form user | [[array,of,movie,ids,being,recommended]]
    nrecommendations = model.recommendForAllUsers(10)

    #Uncomment for per-movie rows
    #nrecommendations = nrecommendations\
    #.withColumn("rec_exp", explode("recommendations"))\
    #.select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))

    nrecommendations.limit(10).show()


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)