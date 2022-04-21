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
    #ratings = spark.read.csv(f'hdfs:/user/el3418/ml-latest-small/ratings.csv', schema='userId INT, movieId INT, rating DOUBLE, timestamp INT')
    ratings = spark.read.option("header",True).csv(f'hdfs:/user/el3418/ml-latest-small/ratings.csv')
    print('ratings.csv schema')
    ratings.printSchema()

    len(list(ratings.select('userId').distinct().toPandas()['userId']))

    #ratings2 = ratings.withColumn("train_val_test", when(col("userId") >=40000 & col("Salary") <= 50000,lit("100")).otherwise(lit("200")))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
