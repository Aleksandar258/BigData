import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import split, regexp_replace, col
from pyspark.ml.functions import vector_to_array
from Services.Repository import Repository
from sklearn.metrics import r2_score


class MachineLearning:
    def __init__(self):
        self.repository = Repository()

    def TrainModel(self, ratio):
        df = self.repository.readFromDatabase()
        df = self.TransformDataset(df)

        indexer = StringIndexer(inputCols=["fueltype", "aspiration", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "CompanyName"],
                                outputCols=["fueltypeI", "aspirationI", "carbodyI", "drivewheelI", "enginelocationI", "enginetypeI","cylindernumberI", "fuelsystemI", "CompanyNameI"])
        df = indexer.fit(df).transform(df)

        featureassembler = VectorAssembler(inputCols=["wheelbase", "carwidth", "curbweight", "enginesize", "boreratio", "horsepower", "citympg",
                                                      "fueltypeI", "aspirationI", "carbodyI", "drivewheelI", "enginelocationI", "enginetypeI",
                                                      "cylindernumberI", "fuelsystemI", "CompanyNameI"], outputCol="Independent Features")
        output = featureassembler.transform(df)

        finalized_data = output.select("Independent Features", "price")

        train_test_split = ratio.split('/')
        train = int(train_test_split[0]) / 100
        test = int(train_test_split[1]) / 100

        train_data, test_data = finalized_data.randomSplit([train, test])

        sc = StandardScaler(inputCol="Independent Features", outputCol="Scaled Independent Features", withMean=True, withStd=True)
        scalerModelTrain = sc.fit(train_data)
        scaledTrainData = scalerModelTrain.transform(train_data)

        scalerModelTest = sc.fit(test_data)
        scaledTestData = scalerModelTest.transform(test_data)

        regressor = LinearRegression(featuresCol='Scaled Independent Features', labelCol='price')
        self.model = regressor.fit(scaledTrainData)
        self.trainD = train_data
        self.testD = test_data
        self.scaledTestD = scaledTestData

    def TestModel(self):
        rez = self.model.evaluate(self.scaledTestD)

        pred = ((rez.predictions.withColumn("xs", vector_to_array("Scaled Independent Features"))).select(["price", "prediction"] + [col("xs")[i] for i in range(16)]))

        y_test = pred.select('price').rdd.flatMap(lambda x: x).collect()
        y_pred = pred.select("prediction").rdd.flatMap(lambda x: x).collect()
        score = r2_score(y_test, y_pred)

        return y_test, y_pred, score

    def TransformDataset(self, df):
        df = df.drop('CarModel', 'carheight', 'stroke', 'compressionratio', 'symboling', 'peakrpm', 'carlength', 'doornumber', 'highwaympg')
        return df

