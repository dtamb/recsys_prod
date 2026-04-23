# stats.py

from pyspark.sql import functions as F
import configs.settings as cfg

def compute_global_mean(ratings_df, rating_col=cfg.RATING_COL):

    mu = ratings_df.select(F.avg(rating_col).alias("mu")).first()["mu"]
    
    return mu