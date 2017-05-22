// Step 1
val loanData = spark.read.
	option("header", "true").
	csv("../prosperLoanData.csv")

loanData.printSchema
loanData.first



// Step 2, choose a few columns

val colName = "ProsperRating (numeric)"
loanData.select(colName).groupBy(colName).count.show

val colName = "EmploymentStatus"
loanData.select(colName).groupBy(colName).count.show

val colName = "IncomeRange"
loanData.select(colName).groupBy(colName).count.show

val colName = "IncomeVerifiable"
loanData.select(colName).groupBy(colName).count.show

val colName = "LoanStatus"
loanData.select(colName).groupBy(colName).count.show



// Step 3, select a few columns

val severalCols = loanData.select("ProsperRating (numeric)", "EmploymentStatus", "IncomeRange", "IncomeVerifiable", "LoanStatus")
severalCols.show


// Step 4, filter out loans with status: "Current" or "Cancelled" 

val interestingCols = severalCols.filter(r => r.getString(4) != "Current" && r.getString(4) != "Cancelled")


// Step 5, convert the columns to numeric 

val convRating = (s: String) => {if (s == null) 0 else s.toInt}

def convEmployment(s: String): Int = {
	val sss = Array("Not available", null, "Not employed", "Other")
	if (sss.contains(s)) 0 else 1
}

def convIncomeRange(s: String): Int = s match {
  case "$0" =>  0
  case "Not employed" =>  0
  case "Not displayed" =>  0
  case "$1-24,999" =>  1 
  case "$25,000-49,999" =>  2
  case "$50,000-74,999" =>  3 
  case "$75,000-99,999" =>  4
  case "$100,000+" =>  5
  case _ => 10000
}

val convIncomeVeri = (b: String) => {if (b == "True") 1 else 0}
def convStatus(s: String): Int = {
	val sss = Array("Defaulted", "Chargedoff", "Past Due")
	if (sss.contains(s)) return 0 else return 1
}

val numericCols = interestingCols.map(r => (convRating(r.getString(0)), convEmployment(r.getString(1)), convIncomeRange(r.getString(2)), convIncomeVeri(r.getString(3)), convStatus(r.getString(4))))

val namedCols = numericCols.toDF("rating", "employment", "income", "incomeVeri", "status")


// Step 6, split data into training data and test data

//val Array(trainData, testData) = namedCols.randomSplit(Array(0.9, 0.1))
//trainData.cache()

// Step 7, form a vector of features

import org.apache.spark.ml.feature.VectorAssembler
val inputCols = namedCols.columns.filter(_ != "status")
val assembler = new VectorAssembler().
  setInputCols(inputCols).
  setOutputCol("features")

val assembledTrainData = assembler.transform(namedCols)  //temp
assembledTrainData.select("features").show(truncate = false)



// Step 8, train a model

import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random

val classifier = new DecisionTreeClassifier().
  setSeed(Random.nextLong()). 
  setLabelCol("status").
  setFeaturesCol("features").
  setPredictionCol("prediction")

val model = classifier.fit(assembledTrainData)
println(model.toDebugString)

model.featureImportances.toArray.zip(inputCols).
  sorted.reverse.foreach(println)

