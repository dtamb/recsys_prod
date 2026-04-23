# stats.py

from pyspark.sql import functions as F
import configs.settings as cfg

def compute_global_mean(ratings_df, rating_col=cfg.RATING_COL):
    '''
    Computes the global average of all ratings in the ratings dataset.
    
    Args:
        ratings_df: Spark DataFrame with ratings data
        rating_col: column to average over
        
    Output:
        mu: global average of ratings
        
    '''
    mu = ratings_df.select(F.avg(rating_col).alias("mu")).first()["mu"]
    
    return mu