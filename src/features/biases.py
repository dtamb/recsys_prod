# biases.py

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
        reg_param: regularisation parameter, typically 5 - 20 (NB: must be same as compute_user_bias reg_param)
        
    Returns:
        item_features_df: original Spark DataFrame with column of item biases
    

    '''
    item_bias = item_features_df.withColumn(
        'item_bias',
        (F.col('item_rating_count')*(F.col('item_avg_rating')-mu))/(reg_param+F.col('item_rating_count'))
    )
        
    return item_bias.select(cfg.ITEM_COL, 'item_bias')

def compute_user_bias(train_df, user_features_df, item_bias_df, mu, reg_param=10):
    
    '''
    Computes user biases for each user with the formula: 
    user_bias = sum(ratings - mu - item_bias) / (delta + num_ratings)
    OR
    user_bias = (num_ratings(user_avg_rating - mu) - sum(item_bias)) / (delta + num_ratings)
    
    Args: 
        train_df: Spark DF with movieID and userID
        user_features_df: Spark DF in format from build_user_features from user_features.py
        item_features_df: Spark DF in format from build_item_features from item_features.py
        mu: global average rating
        reg_param: regularisation parameter, typically 5 - 20 (NB: must be same as compute_item_bias reg_param)
    
    '''
    
    # Select movieID and userID columns only for performance
    train_df = train_df.select(cfg.USER_COL, cfg.ITEM_COL)

    # Join item biases to users via train dataset, then sum item biases per user
    user_sum_item_biases = train_df.join(
        item_bias_df, on=cfg.ITEM_COL, how='left'
    ).groupBy(cfg.USER_COL).agg(
        F.sum(
            F.coalesce(F.col('item_bias'), F.lit(0.0)) # null --> 0 for empty items
        ).alias('sum_item_bias')
    )
    
    
    # Join summed item biases to user features and then compute user biases
    user_bias = user_features_df.join(
        user_sum_item_biases, on=cfg.USER_COL, how='inner'
    ).withColumn(
        'user_bias',
        (
            F.col('user_rating_count')*(F.col('user_avg_rating') - mu) - F.col('sum_item_bias')
        ) / (reg_param + F.col('user_rating_count'))
    )

    return user_bias.select(cfg.USER_COL, 'user_bias')