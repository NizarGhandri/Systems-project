package knn

import org.rogach.scallop._
import org.json4s.jackson.Serialization
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import utils._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val json = opt[String]()
  verify()
}

//case class Rating(user: Int, item: Int, rating: Double)

object Predictor extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  println("")
  println("******************************************************")

  var conf = new Conf(args)
  println("Loading training data from: " + conf.train())
  val trainFile = spark.sparkContext.textFile(conf.train())
  val train = trainFile.map(l => {
      val cols = l.split("\t").map(_.trim)
      Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble)
  })
  assert(train.count == 80000, "Invalid training data")

  println("Loading test data from: " + conf.test())
  val testFile = spark.sparkContext.textFile(conf.test())
  val test = testFile.map(l => {
      val cols = l.split("\t").map(_.trim)
      Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble)
  })
  assert(test.count == 20000, "Invalid test data")

  val values_k = List(10, 30, 50, 100, 200, 300, 400, 800, 942).par

  val results= values_k.map(k => Utils.knn(test, train, k)).toArray

  val RAM = 1.7179860388E10 // 16 GigaBytes

  // Save answers as JSON
  def printToFile(content: String,
                  location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach{
      f => try{
        f.write(content)
      } finally{ f.close }
  }
  conf.json.toOption match {
    case None => ;
    case Some(jsonFile) => {
      var json = "";
      {
        // Limiting the scope of implicit formats with {}
        implicit val formats = org.json4s.DefaultFormats
        val answers: Map[String, Any] = Map(
          "Q3.2.1" -> Map(
            // Discuss the impact of varying k on prediction accuracy on
            // the report.
            "MaeForK=10" -> results(0), // Datatype of answer: Double
            "MaeForK=30" -> results(1), // Datatype of answer: Double
            "MaeForK=50" -> results(2), // Datatype of answer: Double
            "MaeForK=100" -> results(3), // Datatype of answer: Double
            "MaeForK=200" -> results(4), // Datatype of answer: Double
            "MaeForK=300" -> results(5), // Datatype of answer: Double
            "MaeForK=400" -> results(6), // Datatype of answer: Double
            "MaeForK=800" -> results(7), // Datatype of answer: Double
            "MaeForK=942" -> results(8), // Datatype of answer: Double
            "LowestKWithBetterMaeThanBaseline" -> 100, // Datatype of answer: Int
            "LowestKMaeMinusBaselineMae" -> (results(3) - Utils.baseline_MAE) // Datatype of answer: Double
          ),

          "Q3.2.2" ->  Map(
            // Provide the formula the computes the minimum number of bytes required,
            // as a function of the size U in the report.
            "MinNumberOfBytesForK=10" -> 943*10*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=30" -> 943*30*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=50" -> 943*50*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=100" -> 943*100*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=200" -> 943*200*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=300" -> 943*300*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=400" -> 943*400*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=800" -> 943*800*8, // Datatype of answer: Int
            "MinNumberOfBytesForK=942" -> 943*942*8 // Datatype of answer: Int
          ),

          "Q3.2.3" -> Map(
            "SizeOfRamInBytes" -> RAM, // Datatype of answer: Int
            "MaximumNumberOfUsersThatCanFitInRam" -> RAM/(8*3*100) // Datatype of answer: Int
          )

          // Answer the Question 3.2.4 exclusively on the report.
         )
        json = Serialization.writePretty(answers)
      }

      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }

  println("")
  spark.close()
}
