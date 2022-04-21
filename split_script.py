from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct


def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    #####--------------YOUR CODE STARTS HERE--------------#####

    #Use this template to as much as you want for your parquet saving and optimizations!
    links = spark.read.csv(f'hdfs:/user/el3418/ml-latest-small/links.csv')
    print('links.csv schema')
    links.printSchema()

    movies = spark.read.csv(f'hdfs:/user/el3418/ml-latest-small/movies.csv')
    print('movies.csv schema')
    movies.printSchema()

    ratings = spark.read.csv(f'hdfs:/user/el3418/ml-latest-small/ratings.csv')
    print('ratings.csv schema')
    ratings.printSchema()

    tags = spark.read.csv(f'hdfs:/user/el3418/ml-latest-small/tags.csv')
    print('tags.csv schema')
    tags.printSchema()

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
