package utils

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession 
import scala.collection.parallel.ParMap
import scala.collection.parallel.mutable.ParArray





case class Rating(user: Int, item: Int, rating: Double)



object Utils { 

    type UserGroupedDev = ParMap[Int, Map[Int,Double]]



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

    val baseline_MAE: Double = 0.7669

    val l2_norm = (x: Iterable[Double]) => scala.math.sqrt(x.map(scala.math.pow(_, 2)).sum)


    def apu_dev (ratings: RDD[Rating]) = {
        val average_per_user = average_per_attribute("users", ratings)
        val deviations = ratings.map(x => (x.user, x))
                            .join(average_per_user)
                            .mapValues(x => (x._1.item, normalized_deviation(x._1.rating, x._2)))
                            .cache()
        (average_per_user, deviations)
    }


    def preprocess_similarity (ratings: RDD[Rating]) = {
        val (average_per_user, deviations) = apu_dev(ratings)
        lazy val preprocessed_deviations = deviations.groupByKey()
                                                     .mapValues(x => {
                                                        val norm = l2_norm(x.map(_._2))
                                                        x.map(y => (y._1, y._2/norm)).toMap
                                                     })
                                                     .collectAsMap().par 
        (average_per_user, deviations, preprocessed_deviations)
    }

    def compute_similarity (preprocessed_deviations: UserGroupedDev)(user: Int) = {
        def similarity (user1: Map[Int, Double], user2: Map[Int, Double]): Double = {
            (user1.keySet).intersect(user2.keySet).toList.map(item => user1.getOrElse(item, 0.0) * user2.getOrElse(item, 0.0)).sum
        }
        val user_map = preprocessed_deviations.getOrElse(user, Map((0, 0.0)))
        (user, preprocessed_deviations.map(x => (x._1, similarity(user_map, x._2))))
    }
    
    def weighted_average (similarities: ParMap[Int, Double], deviations: Map[Int, Double]): Double = {
        val keys = deviations.keySet.toList.map(user => (user, similarities.getOrElse(user, 0.0)))
        keys.map(u => u._2 * deviations.getOrElse(u._1, 0.0)).sum / keys.map(u => scala.math.abs(u._2)).sum
    }

    def global_dev_similarity (deviations: RDD[(Int, (Int, Double))], similarities: List[(Int, ParMap[Int, Double])]) = {
        deviations.groupBy(_._2._1).flatMap{ case (item, list_users) => {
            val y = list_users.map {case (user, (item, dev)) => (user, dev)}.toMap
            similarities.map(x => ((x._1, item), weighted_average(x._2, y)))
        }}
    }



    def cosine_prediction (test: RDD[Rating], train: RDD[Rating]): Double = {
        val (average_per_user, deviations, preprocessed_deviations) = preprocess_similarity(train)
        val similarities = test.map(_.user).distinct().collect().toList.map(compute_similarity(preprocessed_deviations))
        val items = test.map(_.item).collect().toSet 
        val global_dev = global_dev_similarity(deviations.filter(x => items.contains(x._2._1)), similarities)
        test.map(x => (x.user, x))
            .join(average_per_user)
            .map{case (user, (r, apu)) =>  ((user, r.item), (apu, r.rating))}//.take(10).foreach(println)
            .join(global_dev)
            .mapValues{case ((apu, r), dev) => scala.math.abs(predict(apu, dev) - r)}.values.mean()
    }




    def compute_jaccard(users: ParMap[Int, Set[Int]])(user: Int) = {
        def jaccard(a: Set[Int], b: Set[Int]): Double = {
            val intersection = a & b
            val n = intersection.size.toDouble
            n/(a.size + b.size - n)
        }
        val user_set = users.getOrElse(user, Set())
        (user, users.map(x => (x._1, jaccard(user_set, x._2))))
    } 


    def jaccard_prediction (test: RDD[Rating], train: RDD[Rating]) = {
        val user_sets = train.groupBy(_.user).mapValues(x => x.map(_.user).toSet).collectAsMap().par
        val (average_per_user, deviations) = apu_dev(train)
        val similarities = test.map(_.user).distinct().collect().toList.map(compute_jaccard(user_sets))
        val items = test.map(_.item).collect().toSet 
        val global_dev = global_dev_similarity(deviations.filter(x => items.contains(x._2._1)), similarities)
        test.map(x => (x.user, x))
            .join(average_per_user)
            .map{case (user, (r, apu)) =>  ((user, r.item), (apu, r.rating))}
            .join(global_dev)
            .mapValues{case ((apu, r), dev) => scala.math.abs(predict(apu, dev) - r)}.values.mean()
    }
    

}