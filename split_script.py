from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import when, lit, col

def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    #####--------------YOUR CODE STARTS HERE--------------#####

    #Use this template to as much as you want for your parquet saving and optimizations!
    #schema='userId INT, movieId INT, rating DOUBLE, timestamp INT'
    ratings = spark.read.option("header",True).csv(f'hdfs:/user/el3418/ml-latest-small/ratings.csv')
    print('ratings.csv schema')
    ratings.printSchema()

    #There are 610 unique Ids, but the docs say 600
    print('Number of unique userIds')
    print(len(list(ratings.select('userId').distinct().toPandas()['userId'])))

    #Split uniqueIds based on index slicing
    uniqueIds = list(ratings.select('userId').distinct().toPandas()['userId'])
    train = uniqueIds[:400]
    validation = uniqueIds[400:500]
    test = uniqueIds[500:]

    #Create a new column in the dataframe. Every row will have value 'train' for userId in the train list, 'validation' for userId in the validation list, and 'test' for userId in the test list
    ratings2 = ratings.withColumn("train_val_test", when(col("userId").isin(train),lit("train")).when(col("userId").isin(validation), lit("validation")).otherwise(lit("test")))

    #Per Brian's suggestion in Brightspace discussion tab, training data will have 100% of reviews for users in training, 30% for users in validation, and 30% for users in test
    training_data = ratings2.sampleBy("train_val_test", fractions={'train':1, 'validation': 0.3, 'test': 0.3}, seed=1234)

    #Save to HDFS
    #training_data.write.csv('training_data.csv')

    #Get count of results
    training_data.groupBy('train_val_test').count().orderBy('train_val_test').show()

    ratings.subtract(training_data).show()

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
