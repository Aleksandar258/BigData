import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, regexp_replace, col
from Services.Model.Dataset import Dataset


class Repository:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName('WriteInMySql.com') \
            .config("spark.jars", "mysql-connector-j-8.0.31.jar").getOrCreate()

    def writeCSVInDatabase(self, file):
        df_pyspark = self.spark.read.csv(file, header=True, inferSchema=True)
        df_pyspark = self.transform(df_pyspark)

        try:
            df_pyspark.write \
                .format("jdbc") \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("url", "jdbc:mysql://localhost:3306/testdatabase") \
                .option("dbtable", "cars") \
                .option("user", "root") \
                .option("password", "root") \
                .mode('overwrite').save()

            return True
        except:
            print("An exception occurred")
            return False

    def readFromDatabase(self):
        try:
            df = self.spark.read \
                .format("jdbc") \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("url", "jdbc:mysql://localhost:3306/testdatabase") \
                .option("dbtable", "cars") \
                .option("user", "root") \
                .option("password", "root") \
                .load()

            return df
        except:
            print("An exception occurred")

    def transform(self, df):
        car_name_and_company = split(df['CarName'], ' ')
        df = df.withColumn('CompanyName', car_name_and_company.getItem(0))
        df = df.withColumn('CarModel', car_name_and_company.getItem(1))
        df = df.drop('car_ID', 'CarName')#, 'carheight', 'stroke', 'compressionratio', 'symboling', 'peakrpm', 'carlength', 'doornumber', 'highwaympg')
        df = df.withColumn('CompanyName', regexp_replace('CompanyName', 'maxda', 'mazda'))
        df = df.withColumn('CompanyName', regexp_replace('CompanyName', 'porcshce', 'porsche'))
        df = df.withColumn('CompanyName', regexp_replace('CompanyName', 'vokswagen', 'volkswagen'))
        df = df.withColumn('CompanyName', regexp_replace('CompanyName', 'vw', 'volkswagen'))
        df = df.withColumn('CompanyName', regexp_replace('CompanyName', 'Nissan', 'nissan'))
        df = df.sort(col("CompanyName").asc())
        df = df.na.fill("")
        return df

