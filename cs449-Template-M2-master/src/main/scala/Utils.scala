package utils

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession 
import scala.collection.parallel.ParMap



case class Rating(user: Int, item: Int, rating: Double)



object Utils { 

    type UserGroupedDev = ParMap[Int, Map[Int,Double]]

    val average = (x : RDD[Double]) =>  x.reduce(_ + _) / x.count()


    def average_key_value (rdd: RDD[(Int, (Double, Double))]) = {
        rdd.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).mapValues(x => x._1 / x._2)
    }

    // We use reduceByKey instead of groupby key for performance issues
    def average_per_attribute (att: String, data: RDD[Rating]) = {
        att match {
            case "users" => average_key_value(data.map(x => (x.user, (x.rating, 1))))
            case "items" => average_key_value(data.map(x => (x.item, (x.rating, 1))))
            case _ => throw new IllegalArgumentException("Only accepts keywords users or items")
        }
    }
    
/***********************************************************************************************/

    val scale = (x: Double, avg: Double) => {
        if (x > avg) 5 - avg
        else if (x < avg) avg - 1
        else 1
    }

    val normalized_deviation = (x: Double, avg: Double) => {
        (x - avg)/scale(x, avg)
    }

    def predict (avg_rating_user: Double, avg_deviation_item: Double) = {
        avg_rating_user + avg_deviation_item*scale((avg_rating_user + avg_deviation_item), avg_rating_user)
    }

    val baseline_MAE = 0.7669

    val l2_norm = (x: Iterable[Double]) => scala.math.sqrt(x.map(scala.math.pow(_, 2)).sum)


    def preprocess_similarity (ratings: RDD[Rating]) = {
        val averages_per_user = average_per_attribute("users", ratings)
        val deviations = ratings.map(x => (x.user, x))
                            .join(averages_per_user)
                            .mapValues(x => (x._1.item, normalized_deviation(x._1.rating, x._2)))
                            .cache()
        lazy val preprocessed_deviations = deviations.groupByKey()
                                                     .mapValues(x => {
                                                        val norm = l2_norm(x.map(_._2))
                                                        x.map(y => (y._1, y._2/norm)).toMap
                                                     })
                                                     .collectAsMap().par 
        (averages_per_user, deviations, preprocessed_deviations)
    }

    def compute_similarity (preprocessed_deviations: UserGroupedDev)(user: Int) = {
        def similarity (user1: Map[Int, Double], user2: Map[Int, Double]): Double = {
            (user1.keySet).intersect(user2.keySet).map(item => user1.getOrElse(item, 0.0) * user2.getOrElse(item, 0.0)).sum
        }
        val user_map = preprocessed_deviations.getOrElse(user, Map((0, 0.0)))
        (user, preprocessed_deviations.map(x => (x._1, similarity(user_map, x._2))))
    }
    
    def weighted_average (similarities: ParMap[Int, Double], deviations: Map[Int, Double]): Double = {
        val set = (similarities.keySet).intersect(deviations.keySet).map(user => (user, similarities.getOrElse(user, 0.0)))
        set.map(u => u._2 * deviations.getOrElse(u._1, 0.0)).sum / set.map(u => scala.math.abs(u._2)).sum
    }

    def global_dev_similarity (deviations: RDD[(Int, (Int, Double))], similarities: RDD[(Int, ParMap[Int, Double])]) = {
        //lazy val devs_per_item = deviations.groupBy(_._2._1)
        //devs_per_item.take(1).foreach(println)
        similarities.join(deviations).groupBy(_._2._2._1).mapValues(x => {
            val y = x.map {case (user, (map, (item, dev))) => (user, dev) }.toMap
            x.map {case (user, (map, (item, dev))) => (user, weighted_average(map, y))}
        }).map{case (user, (item,  dev)) => ((user, item), dev)}.cache()
    }

    def cosine_prediction (test: RDD[Rating], train: RDD[Rating]) = {
        val (average_per_user, deviations, preprocessed_deviations) = preprocess_similarity(train)
        val similarities = test.map(_.user).distinct().map(compute_similarity(preprocessed_deviations))
        val global_dev = global_dev_similarity(deviations, similarities)
        test.map(x => (x.user, x))
            .join(average_per_user) 
            .map{case (user, (rating, apu)) =>  ((user, rating.item), apu)}
            .join(global_dev)
            .map{case ((user, rating.item), (apu, dev)) => Rating(user, item, predict(apu, dev))}
        //users.map(compute_similarity(preprocessed_deviations))
        //print(global_dev_similarity(deviations, similarities).getClass)
        /*test.map(x => (x.user, x))
            .join(deviations)
            .join(preprocessed_deviations)*/
        

    }


    


}