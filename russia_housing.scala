

// Step 1a
val econ = spark.read.option("header", "true").csv("../kaggle1/macro.csv")

econ.count        //2484
econ.printSchema
econ.first

// Step 1b
val housing = spark.read.option("header", "true").csv("../kaggle1/train.csv")
housing.count    //30471
housing.printSchema   //a very large number of columns
housing.first

val housingTest = spark.read.option("header", "true").csv("../kaggle1/test.csv")
housingTest.count    //7662
housingTest.printSchema   //one less column (price_doc) than training data
housingTest.first

val subm = spark.read.option("header", "true").csv("../kaggle1/sample_submission.csv")

subm.count        //7662
subm.printSchema
subm.first

//who is ahead the curve. time shift (a couple of years?)

val housingTest = spark.read.option("header", "true").csv("../kaggle1/test.csv")


// Step 2a, look at a few columns of the housing data ============

val colName = "timestamp"
housing.select(colName).groupBy(colName).count.show

val colName = "sub_area"
val x= housing.select(colName).groupBy(colName).count
x.show
x.count  //146

val colName = "life_sq"
val x= housing.select(colName).groupBy(colName).count
x.show
x.count  //176

val colName = "state"
val x= housing.select(colName).groupBy(colName).count
x.show
x.count   //6

val colName = "price_doc"
val x= housing.select(colName).groupBy(colName).count
x.show
x.count  //9296

// Step 2b, ======== Now examine economy data

val colName = "timestamp"
val x= econ.select(colName).groupBy(colName).count
x.show
x.count  //2484

val colName = "cpi"
val x= econ.select(colName).groupBy(colName).count
x.show
x.count  //79, it has NA

x.count 
val colName = "usdrub"
val x= econ.select(colName).groupBy(colName).count
x.show
x.count //1755

val colName = "oil_urals"
val x= econ.select(colName).groupBy(colName).count
x.show
x.count //81

val colName = "rent_price_2room_eco"
val x= econ.select(colName).groupBy(colName).count
x.show
x.count //72

val colName = "rent_price_2room_bus"
val x= econ.select(colName).groupBy(colName).count
x.show
x.count //72

val colName = "gdp_annual"
val x= econ.select(colName).groupBy(colName).count
x.show
x.count  //7



//Step 3a, select a few columns from housing

val housingCols = housing.select("timestamp", "sub_area", "life_sq", "state", "price_doc")
housingCols.show
housingCols.show(10)
housingCols.head(3)

housingCols.orderBy($"price_doc").show(10)


// Step 3b, select a few columns from econ
val econCols = econ.select("timestamp", "cpi", "usdrub", "oil_urals", "rent_price_2room_eco")
econCols.show


// To join the two sets of data, we could use two methods.
// Method 1, first create tables than join through a sql statement
// Step 4a, create tables
housingCols.createOrReplaceTempView("HousingTable") 
econCols.createOrReplaceTempView("EconTable") 

val result = spark.sql("SELECT * FROM HousingTable")
result.show(2)
val result = spark.sql("SELECT * FROM EconTable")
result.show(2)

// Step 4b, left outer join
//val joinType = "left_outer"
//severalCols.join(econCols, "timestamp", joinType).show()


val combined = spark.sql("SELECT * FROM HousingTable LEFT OUTER JOIN EconTable ON HousingTable.timestamp = EconTable.timestamp")

combined.count
combined.show(10)
combined.head(3)

// Method 2 (a better method)
// Step 4
val combined = housingCols.
    join(econCols, Seq("timestamp"), "left_outer").
    select("sub_area", "life_sq", "state", "price_doc", "cpi", "usdrub", "oil_urals", "rent_price_2room_eco")


// Step 5a, filter out data that have NA? 

//val interestingCols = combined.filter(r => r.getString(3) != "NA")
// reduced from 30471 to 16912
// a better way is to change the state to 0





// Step 6, convert string to numeric
def RMSEL(a: Int, p: Int, n: Int): Double = {
	Math.pow((Math.log1p(p) - Math.log1p(a)), 2)
	Math.sqrt
}
def convPrice(s: String): Double = Math.log1p(if (s.contains("NA")) 0 else s.toInt)  // Root Mean Squared Logarithmic Error

def convLifeSq(s: String): Int = {
	val sss = Array("NA")
	if (sss.contains(s)) 0 else s.toInt
}

def convState(s: String): Int = if (Array("NA", "33").contains(s)) 0 else s.toInt


def convCpi(s: String): Double = if (s.contains("NA")) 0 else s.toDouble


def convUsdrub(s: String): Int = {
	val sss = Array("NA")
	if (sss.contains(s)) 0 else s.toInt
}

def convOilUrals(s: String): Int = {
	val sss = Array("NA")
	if (sss.contains(s)) 0 else s.toInt
}

def convRentPrice2roomEco(s: String): Int = {
	val sss = Array("NA")
	if (sss.contains(s)) 0 else s.toInt
}



val convertedCols = combined.map(r => (r.getString(0), convState(r.getString(1)), convState(r.getString(2)), convPrice(r.getString(3)), convCpi(r.getString(4)), convCpi(r.getString(5)), convCpi(r.getString(6)), convCpi(r.getString(7)) )).toDF("sub_area", "life_sq", "state", "price_doc", "cpi", "usdrub", "oil_urals", "rent_price_2room_eco")




// Step 5b, use one hot encoding
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val indexer = new StringIndexer().
setInputCol("sub_area").
setOutputCol("areaIndex").
fit(convertedCols)

val indexed = indexer.transform(convertedCols)
indexed.show

val encoder = new OneHotEncoder().
setInputCol("areaIndex").
setOutputCol("areaVec")

val encoded = encoder.transform(indexed)
encoded.show()


// Step 5c, drop the timestamp column
val namedCols = encoded.drop("sub_area", "areaIndex")
namedCols.show


//val interestingCols = eightCols.map(r => convState(r.getString(0)))
//val x= interestingCols.groupBy("value").count
//x.show


// Step 6, split data into training data and test data

val Array(trainData, testData) = eightCols.randomSplit(Array(0.9, 0.1))
trainData.cache()

// Step 7, form a vector of features

import org.apache.spark.ml.feature.VectorAssembler
val inputCols = namedCols.columns.filter(_ != "price_doc")
val assembler = new VectorAssembler().
  setInputCols(inputCols).
  setOutputCol("features")

val assembledTrainData = assembler.transform(trainData)  
assembledTrainData.select("features").show(truncate = false)

val assembledTestData = assembler.transform(testData)  
assembledTestData.select("features").show(truncate = false)



// Step 8a, train a model, linear regression
import org.apache.spark.ml.regression.LinearRegression
val algLR = new LinearRegression() 
    algLR.setMaxIter(100) 
    algLR.setRegParam(0.3) 
    algLR.setElasticNetParam(0.8) 
    algLR.setLabelCol("price_doc") 
    // 
    val mdlLR = algLR.fit(assembledTrainData) 
    // 
    println(s"Coefficients: ${mdlLR.coefficients} Intercept: ${mdlLR.intercept}") 
    val trSummary = mdlLR.summary 
println(s"numIterations: ${trSummary.totalIterations}") 
println(s"Iteration Summary History: ${trSummary.objectiveHistory.toList}") 
trSummary.residuals.show() 
println(s"RMSE: ${trSummary.rootMeanSquaredError}") 
println(s"r2: ${trSummary.r2}") 


// Step 8b, decision tree
import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random

val classifier = new DecisionTreeClassifier().
  setSeed(Random.nextLong()). 
  setLabelCol("price_doc").
  setFeaturesCol("features").
  setPredictionCol("prediction")

val model = classifier.fit(assembledTrainData)
println(model.toDebugString)

model.featureImportances.toArray.zip(inputCols).
  sorted.reverse.foreach(println)


// Step 9, predict
val predictions = mdlLR.transform(assembledTestData) 
predictions.show() 

// Step 10, evaluate the accuracy
// Calculate RMSE&MSE 
import org.apache.spark.ml.evaluation.RegressionEvaluator
val evaluator = new RegressionEvaluator() 
evaluator.setLabelCol("price_doc") 
val rmse = evaluator.evaluate(predictions) 
println("Root Mean Squared Error = "+"%6.3f".format(rmse)) 
// 
evaluator.setMetricName("mse") 
val mse = evaluator.evaluate(predictions) 
println("Mean Squared Error = "+"%6.3f".format(mse)) 
// 

