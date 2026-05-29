# ranking_df.py

import configs.settings as cfg
from pyspark.sql import functions as F

def create_ranking_df(similarity_score, user_feature, item_feature, user_bias, item_bias):
    '''
    Merges all features into a single dataframe for ranking 
    
    Args:
    
    
    Returns:
    
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