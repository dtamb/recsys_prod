# ranking_df.py

import configs.settings as cfg
from pyspark.sql import functions as F

def create_ranking_df(similarity_score, user_feature, item_feature, user_bias, item_bias):
    '''
    Merges all features into a single dataframe for ranking.
    
    ranking DF schema
    root
     |-- movieId: integer (nullable = true)
     |-- userId: integer (nullable = true)
     |-- als_score: float (nullable = true)
     |-- similarity: double (nullable = true)
     |-- user_avg_rating: double (nullable = true)
     |-- user_rating_std: double (nullable = false)
     |-- user_log_rating_count: double (nullable = true)
     |-- days_since_last_activity: integer (nullable = true)
     |-- user_bayes_std: double (nullable = true)
     |-- item_avg_rating: double (nullable = true)
     |-- item_bayesian_avg: double (nullable = true)
     |-- item_log_rating_count: double (nullable = true)
     |-- user_bias: double (nullable = true)
     |-- item_bias: double (nullable = true)
    
    Args:
        similarity_score --> ranking.tag_similarity : compute_tag_similarity
        user_feature --> features.user_features : build_user_features
        item_feature --> features.item_features : build_item_features
        user_bias --> features.biases : compute_user_bias
        item_bias --> features.biases : compuute_item_bias
    
    Returns:
        ranking: Spark DF with full set of features for ranking
    
    '''
    
    user_feature = user_feature.drop('user_rating_count', 'user_std')
    item_feature = item_feature.drop('item_rating_count')
    
    ranking = similarity_score.join(
        user_feature, on=cfg.USER_COL, how='inner'
    ).join(
        item_feature, on=cfg.ITEM_COL, how='inner'
    ).join(
        user_bias, on=cfg.USER_COL, how='inner'
    ).join(
        item_bias, on=cfg.ITEM_COL, how='inner'
    )
    
    return ranking