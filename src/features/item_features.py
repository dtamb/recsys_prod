# item_features.py

from pyspark.sql import functions as F
import configs.settings as cfg

def build_item_features(ratings_df, mu, C=50):
    '''
    Creates a dataframe of averages, bayesian averages and log(1+ranking count)
    over each item in the dataframe.
    
    Args:
        ratings_df: Spark Dataframe of ratings by movie and item
        mu: average global rating
        C: confidence number for tuning
        
    Returns:
        items_features: Spark Dataframe of rating average, bayesian rating average
        and log(1 + ranking count) for each item_id 
    
    '''
    
    # Creating item columns of average rating, rating sum and rating count
    item_features = ratings_df.groupBy(cfg.ITEM_COL).agg(
        F.avg(cfg.RATING_COL).alias('item_avg_rating'),
        F.count(cfg.RATING_COL).alias('item_rating_count')
    )
    
    # Bayesian Average
    # item_i_bay_avg = (C * mu + item_i_sum)/(C + item_i_num_ratings)
    # where mu = global average, C = confidence number
    # 
    # NB: item_i_sum = item_i_avg * item_i_num_ratings
    
    item_features = item_features.withColumn(
        'item_bayesian_avg',
        (F.lit(C)*F.lit(mu) + F.col('item_avg_rating')*F.col('item_rating_count'))/(F.lit(C) + F.col('item_rating_count'))
    ).withColumn(
        'log_rating_count',
        F.log1p(F.col('item_rating_count'))
    )
    
    return item_features.drop('item_rating_count')
