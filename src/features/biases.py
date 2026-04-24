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
    
    Returns:
        user_bias: Spark DF with userID and user bias columns
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

def compute_expected_rating(train_df, user_bias, item_bias, mu):
    '''
    Computes expected rating with the formula:
        exp_rat = mu + user_bias + item_bias
    
    Args:
        train_df: Spark DF with userID, movieID and ratings columns
        user_bias: Spark DF with userID and user_bias columns
        item_bias: Spark DF with itemID and item_bias columns
        mu: global average rating
    
    Returns:
        expected rating: userID, itemID, rating, expected rating columns
    '''
    
    # Drop timestamp data
    train_df = train_df.drop(cfg.TIMESTAMP_COL)
    
    # Join user and item biases with ratings via train data
    expected_rating = train_df.join(
        user_bias, on=cfg.USER_COL, how='inner'
    ).join(
        item_bias, on=cfg.ITEM_COL, how='inner'
    )
    
    # Compute expected rating
    expected_rating = expected_rating.withColumn(
        'expected_rating',
        mu+F.col('user_bias')+F.col('item_bias')
    )
    
    return expected_rating

def compute_user_weights(expected_rating, user_feature, tau=1, epsilon=1e-6):
    '''
    User weights for each film to use on movie tags and compute a representative user tag for each user.
    
    weight = tanh( (rating - expected_rating) / (tau*user_bayes_std + epsilon) )
       where tau: variable to control smoothing, 
           epsilon: error correction to avoid dividing by 0
           tanh: used to smooth and bound weights to (-1,1)
           
    
    Args: 
        expected_rating: Spark DF from compute_expected_rating function (userID, movieID, rating, expected_rating)
        user_feature: Spark DF from features/user_features.py (userID, user_bayes_std)
        tau: smoothing variable, typically from 0.5 to 3
        epsilon: error correction for zero-value user_bayes_std 
        
    Returns:
        weighting: Spark DF with userID, movieID, weight
            weight is between (-1,1) and represents how significant the film represents the user as a function
                of the users rating variability (user_bayes_std), their rating and expected rating
    
    '''
    # Select required columns from expected_rating and user_feature
    expected_rating = expected_rating.select(cfg.USER_COL, cfg.ITEM_COL, cfg.RATING_COL,'expected_rating')
    user_feature = user_feature.select(cfg.USER_COL, 'user_bayes_std')
    
    # Join dataframes and calculate weightings
    weighting = expected_rating.join(
        user_feature, on=cfg.USER_COL, how='inner'
    ).withColumn(
        'weight',
        F.tanh(
            (F.col(cfg.RATING_COL) - F.col('expected_rating'))/(tau * F.col('user_bayes_std') + epsilon)
        )
    )
    
    return weighting.select(cfg.USER_COL, cfg.ITEM_COL, 'weight')
    