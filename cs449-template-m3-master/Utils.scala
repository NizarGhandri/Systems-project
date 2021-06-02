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

    val l2_norm = (m: CSCMatrix) => (m *:* m)*reducer(m.rows).mapActiveValues(scala.math.sqrt) 

    def average_per_user(train: CSCMatrix) =  {
        val reduce = reducer(train.cols)
        val sizes = train.mapActiveValues(1)
        (train * reducer) /:/ (sizes * reducer)
    }

    def deviations(train: CSCMatrix, averages: CSCMatrix) = {
        train.mapActivePairs(((row, _), v) => normalized_deviation(v, averages(row)))
    }

    def preprocess_similarity(train: CSCMatrix) = {
        val apu = average_per_user(train)
        val dev = deviations(train, apu)
        val preprocessed_deviations = m /:/ l2_norm(dev)
        (apu, dev, preprocessed_deviations)
    }

    def compute_similarity(preprocessed: CSCMatrix, k: Int) = {
        val all_sims = preprocessed*preprocessed.t
        val u = preprocessed.rows
        val builder = new CSCMatrix.Builder[Double](rows=u, cols=k)

        for (i <- 0 until u) {
            for (j <- argtopk(all_sims(u, ::), k)){
                builder.add(i, j, all_sims(i, j))
            }
        }

        builder.result()

    }

    def knn(test: CSCMatrix, train: CSCMatrix, k: Int): Double = {
        val (average_per_user, deviations, preprocessed_deviations) = preprocess_similarity(train)
        val similarities = test.map(_.user).distinct().collect().toList.map(compute_similarity_knn(preprocessed_deviations, k))
        val items = test.map(_.item).collect().toSet 
        val global_dev = global_dev_similarity(deviations.filter(x => items.contains(x._2._1)), similarities)
        test.map(x => (x.user, x))
            .join(average_per_user)
            .map{case (user, (r, apu)) =>  ((user, r.item), (apu, r.rating))}
            .leftOuterJoin(global_dev)
            .mapValues{case ((apu, r), dev) => scala.math.abs(predict(apu, dev.getOrElse(0.0)) - r)}.values.mean()
        
    }










    /**********************************************************************************************************************************************/

    def compute_similarity(preprocessed: CSCMatrix) = {
        val u = preprocessed.rows
        val builder = new CSCMatrix.Builder[Double](rows=u, cols=u)
        for ((k,v) <- preprocessed.activeIterator) {
            val row = k._1
            val col = k._2
            builder.add(row, col, preprocessed(row, :) * preprocessed(:, col))
        }
        builder.result()
    }


    
    

}