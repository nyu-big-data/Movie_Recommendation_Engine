from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, mean, collect_list

def main(spark):

    train = spark.read.option("header", False).csv(f'hdfs:/user/el3418/training_data.csv')
    train = train.toDF('userId', 'movieId', 'rating', 'timestamp', 'split')
    train = train.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', train['rating'].cast('float'))

    val = spark.read.option("header", False).csv(f'hdfs:/user/el3418/validation_data.csv')
    val = val.toDF('userId', 'movieId', 'rating', 'timestamp', 'split')
    val = val.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', val['rating'].cast('float'))

    test = spark.read.option("header", False).csv(f'hdfs:/user/el3418/test_data.csv')
    test = test.toDF('userId', 'movieId', 'rating', 'timestamp', 'split')
    test = test.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', test['rating'].cast('float'))

    als = ALS(
        maxIter=3,
        userCol="userId", 
        itemCol="movieId",
        ratingCol="rating", 
        nonnegative = True, 
        implicitPrefs = False,
        coldStartStrategy="drop"
    )

    # Add hyperparameters and their respective values to param_grid
    param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [10, 50, 100, 150]) \
                .addGrid(als.regParam, [.01, .05, .1, .15]) \
                .build()
            
    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    #Fit cross validator to the 'train' dataset
    model = cv.fit(train)

    #Extract best model from the cv model above
    best_model = model.bestModel

    # Print best_model
    print(type(best_model))

    # Complete the code below to extract the ALS model parameters
    print("**Best Model**")

    # # Print "Rank"
    print("  Rank:", best_model._java_obj.parent().getRank())

    # Print "MaxIter"
    print("  MaxIter:", best_model._java_obj.parent().getMaxIter())

    # Print "RegParam"
    print("  RegParam:", best_model._java_obj.parent().getRegParam())

    # View the predictions
    test_predictions = best_model.transform(test)
    RMSE = evaluator.evaluate(test_predictions)
    print(RMSE)

    #Output in the form user | [[array,of,movie,ids,being,recommended]]
    nrecommendations = best_model.recommendForAllUsers(100)

    #Uncomment for per-movie rows
    #nrecommendations = nrecommendations\
    #.withColumn("rec_exp", explode("recommendations"))\
    #.select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))

    #Transform data into form suitable for RankingMetrics
    movies_list = test.groupBy("userId", "movieId")\
        .agg(mean('rating').alias('avg_rating'))\
            .groupBy("userId").agg(collect_list("movieId")\
                .alias("movies_list"))

    nrecommendations = nrecommendations.withColumn("recommendations", col("recommendations").getField("movieId"))
    nrecommendations.limit(10).show()
    movies_list.limit(10).show()
    joined_col = movies_list.join(nrecommendations, ["userId"])
    dropped_cols = joined_col.drop("userId")
    
    # define a list
    l=[]

    # convert to tuple and append to list
    for i in dropped_cols.collect():
        l.append(tuple(i))

    predictionAndLabels = spark.sparkContext.parallelize(l)
    metrics = RankingMetrics(predictionAndLabels)
    print(metrics.meanAveragePrecisionAt(100))


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)