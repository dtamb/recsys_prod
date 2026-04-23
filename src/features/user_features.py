# user_features.py

from pyspark.sql import functions as F
import configs.settings as cfg

def build_user_features(ratings_df):
    
    '''
    Converts a ratings dataframe (train data) into user_features for content filtering.
    
    Args:
        ratings_df: Spark ratings DataFrame (MovieLens20M)
        
    Returns:
        user_features: Spark DataFrame with the following features:
            user_avg_rating
            log_rating_count: log(1 + rating count) to avoid nulls from new
                users
           days_since_last_activity: calculated from latest timestamp in
               dataset
    '''
    
    # Calculating max timestamp of df to use as "today"
    max_timestamp = ratings_df.select(
        F.max(cfg.TIMESTAMP_COL).alias('max_ts')
    ).collect()[0]['max_ts']
    
    # Creating user columns of average rating, rating count and last activity
    user_features = ratings_df.groupBy(cfg.USER_COL).agg(
        F.avg(cfg.RATING_COL).alias('user_avg_rating'),
        F.count(cfg.RATING_COL).alias('user_rating_count'),
        F.max(cfg.TIMESTAMP_COL).alias('last_activity')
    )
    
    # Creating user features of log(rating_count + 1) and days since last activity
    user_features = user_features.withColumn(
        'log_rating_count',
        F.log1p(F.col('user_rating_count'))
    ).withColumn(
        'days_since_last_activity',
        F.datediff(F.lit(max_timestamp), F.col('last_activity'))
    )
    
    return user_features.drop('last_activity') 
    