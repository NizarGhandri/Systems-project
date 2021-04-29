package similarity

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

  val NUMBER_OF_EXEC = 5

  val (cosine_MAE, similarities) = Utils.cosine_prediction(test, train)
  val user_sets = train.groupBy(_.user).mapValues(x => x.map(_.item).toSet).collectAsMap().par
  val sim_mult = test.map(_.user).distinct().flatMap(x => {
    val u_set = user_sets.getOrElse(x, Set())
    user_sets.map(y => (u_set intersect y._2).size.toDouble).toList
    }).collect.toList

  val average_similarity = sim_mult.sum / sim_mult.size

  val jaccard_MAE = Utils.jaccard_prediction(test, train)
  val time_for_preds = Utils.measure_performance(spark, conf.train(), conf.test(), Nil, NUMBER_OF_EXEC,
                                                (test: RDD[Rating], train: RDD[Rating]) => {Utils.cosine_prediction(test, train)}
                                                )
  val average_time_for_preds = time_for_preds.sum / time_for_preds.size
  val time_for_similarity = Utils.measure_performance(spark, conf.train(), conf.test(), Nil, NUMBER_OF_EXEC, 
                                                (test: RDD[Rating], train: RDD[Rating]) => {
                                                  val (average_per_user, deviations, preprocessed_deviations) = Utils.preprocess_similarity(train)
                                                  test.map(_.user).distinct().collect().toList.map(Utils.compute_similarity(preprocessed_deviations))
                                                })
  val average_time_for_similarity = time_for_similarity.sum / time_for_similarity.size

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
          "Q2.3.1" -> Map(
            "CosineBasedMae" -> cosine_MAE, // Datatype of answer: Double
            "CosineMinusBaselineDifference" -> (cosine_MAE - Utils.baseline_MAE) // Datatype of answer: Double
          ),

          "Q2.3.2" -> Map(
            "JaccardMae" -> jaccard_MAE, // Datatype of answer: Double
            "JaccardMinusCosineDifference" -> (jaccard_MAE - cosine_MAE) // Datatype of answer: Double
          ),

          "Q2.3.3" -> Map(
            // Provide the formula that computes the number of similarity computations
            // as a function of U in the report.
            "NumberOfSimilarityComputationsForU1BaseDataset" ->  similarities.size * similarities.head._2.size // Datatype of answer: Int
          ),

          "Q2.3.4" -> Map(
            "CosineSimilarityStatistics" -> Map(
              "min" -> sim_mult.min,  // Datatype of answer: Double
              "max" -> sim_mult.max, // Datatype of answer: Double
              "average" -> average_similarity, // Datatype of answer: Double
              "stddev" -> Utils.stddev(sim_mult, average_similarity) // Datatype of answer: Double
            )
          ),

          "Q2.3.5" -> Map(
            // Provide the formula that computes the amount of memory for storing all S(u,v)
            // as a function of U in the report.
            "TotalBytesToStoreNonZeroSimilarityComputationsForU1BaseDataset" -> similarities.flatMap(x => x._2.toList).filter(x => x._2 !=0).size * 8  // Datatype of answer: Int
          ),

          "Q2.3.6" -> Map(
            "DurationInMicrosecForComputingPredictions" -> Map(
              "min" -> time_for_preds.min,  // Datatype of answer: Double
              "max" -> time_for_preds.max, // Datatype of answer: Double
              "average" -> average_time_for_preds, // Datatype of answer: Double
              "stddev" -> Utils.stddev(time_for_preds, average_time_for_preds) // Datatype of answer: Double
            )
            // Discuss about the time difference between the similarity method and the methods
            // from milestone 1 in the report.
          ),

          "Q2.3.7" -> Map(
            "DurationInMicrosecForComputingSimilarities" -> Map(
              "min" -> time_for_similarity.min,  // Datatype of answer: Double
              "max" -> time_for_similarity.max, // Datatype of answer: Double
              "average" -> average_time_for_similarity, // Datatype of answer: Double
              "stddev" -> Utils.stddev(time_for_similarity, average_time_for_similarity) // Datatype of answer: Double
            ),
            "AverageTimeInMicrosecPerSuv" -> average_time_for_similarity / (similarities.size * similarities.head._2.size), // Datatype of answer: Double
            "RatioBetweenTimeToComputeSimilarityOverTimeToPredict" -> average_time_for_similarity/average_time_for_preds // Datatype of answer: Double
          )
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
