# biases.py

# +-------+------------------+-----------------+------------------+------------------+
# |movieId|   item_avg_rating|item_rating_count| item_bayesian_avg|  log_rating_count

from pyspark.sql import functions as F
import configs.settings as cfg

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
    item_bias = item_features_df.withColumn(
        'item_bias',
        (F.col('item_rating_count')*(F.col('item_avg_rating')-mu))/(reg_param+F.col('item_rating_count'))
    )
        
    return item_bias

def compute_user_bias(train_df, user_features_df, item_bias_df, mu, delta=10):
    
    '''
    Computes user biases for each user with the formula: 
    user_bias = sum(ratings - mu - item_bias) / (delta + num_ratings)
    OR
    user_bias = (num_ratings(user_avg_rating - mu) - sum(item_bias)) / (delta + num_ratings)
    
    '''

    # Join item biases to users via train dataset, then sum item biases per user
    user_sum_item_biases = train_df.join(
        item_bias_df, on=cfg.ITEM_COL, how='inner'
    ).groupBy(cfg.USER_COL).agg(
        F.sum('item_bias').alias('sum_item_bias')
    )
    
    # Join summed item biases to user features and then compute user biases
    user_bias = user_features_df.join(
        user_sum_item_biases, on=cfg.USER_COL, how='inner'
    ).withColumn(
        'user_bias',
        (
            F.col('user_rating_count')*(F.col('user_avg_rating') - mu) - F.col('sum_item_bias')
        ) / (delta + F.col('user_rating_count'))
    )

    return user_bias.drop('sum_item_bias')