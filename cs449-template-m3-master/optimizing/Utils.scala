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

    def multiply_iter (a: CSCMatrix[Double], b: CSCMatrix[Double]) = {
        val n = a.cols
        val m = b.rows
        assert(n == m)
        val b_tr = b.t
        val builder = new CSCMatrix.Builder[Double](rows=a.rows, cols=b_tr.rows)
        for (i <- 0 until a.rows) {
            for (j <- 0 until b_tr.rows) {
                builder.add(i, j, a(i, 0 until n).t.toDenseVector.t * b_tr(i, 0 until m).t.toDenseVector)
            }
        }
        builder.result()
    }

    val l2_norm = (m: CSCMatrix[Double]) => ((m *:* m)*reducer(m.cols)).mapValues(scala.math.sqrt)
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
        val all_sims = preprocessed*preprocessed.t//multiply_iter(preprocessed, preprocessed.t) 
        val users = preprocessed.rows
        val builder = new CSCMatrix.Builder[Double](rows=users, cols=users)
        for (i <- 0 until users) {
            for (j <- argtopk(all_sims(i, 0 until users).t, k+1)){
                if (j != i) builder.add(i, j, all_sims(i, j))
            }
        }
        builder.result()
    }

    def global_dev(deviations: CSCMatrix[Double], similarities: CSCMatrix[Double]) = {
        val dev_transposed = deviations.t //for data locality
        //val similarities_per_item = similarities *:* deviations.mapActiveValues(x => 1.0) 
        val r = similarities.rows
        val c = deviations.cols
        val l1 = similarities.mapActiveValues(_.abs) * deviations.mapActiveValues(x => 1.0)
        val users = similarities.rows
        //val l1_sim = l1_norm(adj)
        val builder = new CSCMatrix.Builder[Double](rows=r, cols=c)
        for (i <- 0 until c) {
            val adj = similarities * dev_transposed(i, 0 until r).t.toDenseVector  
            for (k <- 0 until r) {
                //val l1 = deviations(0 until users, col).mapActiveValues(_.abs).toDenseVector.t :*:  simiarities(row) 
                val v = adj(k)/l1(k, i)
                if (!v.isNaN) builder.add(k, i, v)   
            }
        }
        builder.result()
    }

    def predict (test: CSCMatrix[Double], apu: DenseVector[Double], global_dev: CSCMatrix[Double]) = {
        val builder = new CSCMatrix.Builder[Double](rows=test.rows, cols=test.cols)
         for ((k,v) <- test.activeIterator) {
            val row = k._1
            val col = k._2
            builder.add(row, col, predict_single(apu(row), global_dev(row, col)))   
        }
        builder.result()
    }

    def predict_single (avg_rating_user: Double, avg_deviation_item: Double) = {
        avg_rating_user + avg_deviation_item*scale((avg_rating_user + avg_deviation_item), avg_rating_user)
    }

    def knn(test: CSCMatrix[Double], train: CSCMatrix[Double], k: Int) = {
        val (average_per_user, deviations, preprocessed_deviations) = preprocess_similarity(train)
        val similarities = compute_similarity(preprocessed_deviations, k)
        val gdev = global_dev(deviations, similarities)
        predict(test, average_per_user, gdev)
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


    def measure_performance ( acc: List[Double], number_of_exec: Int, f: () => Any): List[Double] = {
        if (number_of_exec == 0) acc
        else {
            val t = System.nanoTime()
            f()
            val delta = System.nanoTime() - t
            measure_performance(nano_to_micro(delta)::acc, number_of_exec-1, f)
        }
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