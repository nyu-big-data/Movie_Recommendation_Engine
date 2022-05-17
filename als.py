from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, mean, collect_list
import time
from sklearn.metrics import average_precision_score

def mAP(top_k_recommendations, user_labels):
    ap = 0
    for movie_id in user_labels:
        running_sum = 0
        for rank, r in enumerate(top_k_recommendations):
            if r == movie_id:
                running_sum = 1 + 1/(rank+1)
                break 
        ap += (running_sum/len(top_k_recommendations))
    
    return ap/len(user_labels)

def main(spark, crossValidate=False):

    train = spark.read.option("header", True).csv(f'hdfs:/user/el3418/new_train_large.csv')
    train = train.toDF('userId', 'movieId', 'rating', 'timestamp')
    train = train.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', col('rating').cast('float'))

    val = spark.read.option("header", True).csv(f'hdfs:/user/el3418/new_val_large.csv')
    val = val.toDF('userId', 'movieId', 'rating', 'timestamp')
    val = val.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', col('rating').cast('float'))

    test = spark.read.option("header", True).csv(f'hdfs:/user/el3418/new_test_large.csv')
    test = test.toDF('userId', 'movieId', 'rating', 'timestamp')
    test = test.withColumn('userId', col('userId').cast('integer')).withColumn('movieId', col('movieId').cast('integer')).withColumn('rating', col('rating').cast('float'))

    als = ALS(
        rank=100,
        maxIter=3,
        regParam=0.15,
        userCol="userId", 
        itemCol="movieId",
        ratingCol="rating", 
        nonnegative = True, 
        implicitPrefs = False,
        coldStartStrategy="drop"
    )

    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    if(crossValidate):
        # Add hyperparameters and their respective values to param_grid
        param_grid = ParamGridBuilder() \
                    .addGrid(als.rank, [10, 50, 100, 150]) \
                    .addGrid(als.regParam, [.01, .05, .1, .15]) \
                    .build()

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

        test_predictions = best_model.transform(test)

        nrecommendations = best_model.recommendForAllUsers(100)
    else:
        start = time.process_time()
        model = als.fit(train)
        end = time.process_time()
        elapsed_time = end - start
        print("Time to fit model: ", elapsed_time)

        start = time.process_time()
        test_predictions = model.transform(test)

        #Output in the form user | [[array,of,movie,ids,being,recommended]]
        nrecommendations = model.recommendForAllUsers(100)
        end = time.process_time()
        elapsed_time = end - start
        print("Time to transform and give recommendations: ", elapsed_time)
            
    #test_predictions = model.transform(test)
    start = time.process_time()
    RMSE = evaluator.evaluate(test_predictions)
    end = time.process_time()
    elapsed_time = end - start
    print("Time to calculate RMSE: ", elapsed_time)
    print(RMSE)

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

    dropped_cols.show()
    
    rdd=dropped_cols.rdd.map(lambda x: 
        (x[0],x[1],mAP(x[0],x[1]))
        ) 
    result_df = rdd.toDF(["movies_list", "recommendations", "precision"])
    
    print(result_df.select(mean('precision')).collect())

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als').getOrCreate()
    crossValidate = False

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark, crossValidate)