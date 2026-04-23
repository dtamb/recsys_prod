# biases.py

# +-------+------------------+-----------------+------------------+------------------+
# |movieId|   item_avg_rating|item_rating_count| item_bayesian_avg|  log_rating_count

from pyspark.sql import functions as F

def compute_item_bias(item_features_df, mu, reg_param=10):

    '''
    Computes item biases for each item with the formula: 
        item_bias = sum(ratings - mu) / (delta + num_ratings)
    NB: sum of ratings --> num_ratings*item_avg_rating, 
    i.e. numerator ==> num_ratings * (item_avg_rating - mu)
    
    Args:
        item_features_df: Spark DataFrame with item_rating_count, item_avg_rating
        mu: global ratings mean
        reg_param: regularisation parameter, typically 5 - 20
        
    Returns:
        item_features_df: original Spark DataFrame with column of item biases
    

    '''
    item_features_df = item_features_df.withColumn(
        'item_bias',
        (F.col('item_rating_count')*(F.col('item_avg_rating')-mu))/(reg_param+F.col('item_rating_count'))
    )
        
    return item_features_df  