package utils

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession 
import scala.collection.parallel.ParMap
import scala.collection.parallel.mutable.ParArray


import breeze.linalg._
import breeze.numerics._


object Utils {

    val reducer = (n: Int) => DenseVector.ones[Double](n) 

    val scale = (x: Double, avg: Double) => {
        if (x > avg) 5 - avg
        else if (x < avg) avg - 1
        else 1
    } 

    val normalized_deviation = (x: Double, avg: Double) => {
        (x - avg)/scale(x, avg)
    }

    val l2_norm = (m: CSCMatrix[Double]) => (m.mapActiveValues(x => x*x)*reducer(m.cols)).mapValues(scala.math.sqrt)
    val l1_norm = (m: CSCMatrix[Double]) =>  m.mapActiveValues(_.abs)*reducer(m.cols)

    def average_per_user(train: CSCMatrix[Double]) =  {
        val reduce = reducer(train.cols)
        val sizes = train.mapActiveValues(x => 1.0)
        (train * reduce) /:/ (sizes * reduce)
    }

    def deviations(train: CSCMatrix[Double], averages: DenseVector[Double]) = {
        val builder = new CSCMatrix.Builder[Double](rows=train.rows, cols=train.cols)
        for ((k,v) <- train.activeIterator) {
            val row = k._1
            val col = k._2
            builder.add(row, col, normalized_deviation(v, averages(row)))   
        }
        builder.result()
    }

    def preprocess_similarity(train: CSCMatrix[Double]) = {
        val apu = average_per_user(train)
        val dev = deviations(train, apu)
        val l2_dev = l2_norm(dev)
        val builder = new CSCMatrix.Builder[Double](rows=dev.rows, cols=dev.cols)
        for ((k,v) <- dev.activeIterator) {
            val row = k._1
            val col = k._2
            builder.add(row, col, v/l2_dev(row))   
        }
        
        (apu, dev, builder.result())
    }

    def compute_similarity(preprocessed: CSCMatrix[Double], k: Int) = {
        val users = preprocessed.rows
        val items = preprocessed.cols
        val builder = new CSCMatrix.Builder[Double](rows=users, cols=users)
        for (i <- 0 until users) {
            val all_sims = preprocessed * preprocessed(i, 0 until items).t.toDenseVector
            for (j <- argtopk(all_sims, k+1)){
                if (j != i) builder.add(i, j, all_sims(j))
            }
        }
        builder.result()
    }

    def global_dev(deviations: CSCMatrix[Double], similarities: CSCMatrix[Double]) = {
        val r = similarities.rows
        val c = deviations.cols
        val l1 = similarities.mapActiveValues(_.abs)
        val builder = new CSCMatrix.Builder[Double](rows=r, cols=c)
        for (i <- 0 until c) {
            val vect = deviations(0 until r, i).toDenseVector 
            val adj = (similarities * vect) /:/  (l1 * vect.map(x => if(x == 0.0) 0.0 else 1.0))
            for (k <- 0 until r) {
                val v = adj(k)
                if (!v.isNaN) builder.add(k, i, v)   
            }
        }
        builder.result()
    }

    def predict (test: CSCMatrix[Double], apu: DenseVector[Double], global_dev: CSCMatrix[Double], global_average: Double) = {
        val builder = new CSCMatrix.Builder[Double](rows=test.rows, cols=test.cols)
         for ((k,v) <- test.activeIterator) {
            val row = k._1
            val col = k._2
            val pred = predict_single(apu(row), global_dev(row, col))
            builder.add(row, col, if(pred == 0.0) global_average else pred)   
        }
        builder.result()
    }

    def predict_single (avg_rating_user: Double, avg_deviation_item: Double) = {
        avg_rating_user + avg_deviation_item*scale((avg_rating_user + avg_deviation_item), avg_rating_user)
    }

    def sim_dev_apu (test: CSCMatrix[Double], train: CSCMatrix[Double], k: Int) = {
        val (average_per_user, deviations, preprocessed_deviations) = preprocess_similarity(train)
        val similarities = compute_similarity(preprocessed_deviations, k)
        (similarities, deviations, average_per_user)
    }

    def knn(test: CSCMatrix[Double], train: CSCMatrix[Double], k: Int, global_average: Double) = {
        val (similarities, deviations, average_per_user) = sim_dev_apu(test, train, k)
        predict(test, average_per_user, global_dev(deviations, similarities), global_average)
    }

    def mae (test: CSCMatrix[Double], pred: CSCMatrix[Double]) = {
        var sum = 0.0
        for ((_,v) <- (test-pred).mapActiveValues(_.abs).activeIterator) {
            sum += v  
        }
        sum/test.activeSize
    }


    //  def reload_data (spark: org.apache.spark.sql.SparkSession, path_train: String, path_test: String): (RDD[Rating], RDD[Rating]) = {
    //     val trainFile = spark.sparkContext.textFile(path_train)
    //     val train = trainFile.map(l => {
    //         val cols = l.split("\t").map(_.trim)
    //         Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble)
    //     }) 
    //     val testFile = spark.sparkContext.textFile(path_test)
    //     val test = testFile.map(l => {
    //         val cols = l.split("\t").map(_.trim)
    //         Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble)
    //     }) 
    //     (train, test)
    // }

    def measure (f: () => Any, x: Int)= {
        println(x)
        val t = System.nanoTime()
        f()
        nano_to_micro(System.nanoTime() - t)
    }

    def measure_performance (number_of_exec: Int, f: () => Any) = {
        (0 until number_of_exec).toList.map(x => measure(f, x))
        //for (i <- 0 until number_of_exec) yield nano_to_micro(measure(f))
    }


    val nano_to_micro = (x: Long) => x/(scala.math.pow(10, 3))

    val stddev = (l: List[Double], avg: Double) => scala.math.sqrt(l.map(x => scala.math.pow((x - avg), 2)).sum / l.size)





    /**********************************************************************************************************************************************/

    // def compute_similarity(preprocessed: CSCMatrix[Double]) = {
    //     val u = preprocessed.rows
    //     val builder = new CSCMatrix.Builder[Double](rows=u, cols=u)
    //     for ((k,v) <- preprocessed.activeIterator) {
    //         val row = k._1
    //         val col = k._2
    //         builder.add(row, col, preprocessed(row, :) * preprocessed(:, col))
    //     }
    //     builder.result()
    // }


    
    

}