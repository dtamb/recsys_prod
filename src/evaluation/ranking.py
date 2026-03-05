# ranking.py

from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics as RM

def prepare_eval_df(recs_df, test_df, user_col, item_col):
    '''
    Args:
        - recs_df: recommendations per userid, itemid and rating
        - test_df: filtered for relevant items, e.g. ratings >=4
        - user_col
        - item_col

    Output: 
        - eval_df: grouped df of predictions and thresholded test items per user

    '''
    # Group predictions per user
    pred_grouped = recs_df.groupBy(user_col).agg(
        F.collect_list(item_col).alias('pred_items')
    )

    true_grouped = test_df.groupBy(user_col).agg(
        F.collect_list(item_col).alias('true_items')
    )
    # Join to create eval df
    return pred_grouped.join(true_grouped, on=user_col)

def compute_ranking_metrics(eval_df, k):
    '''
    Args: 
        - eval_df: grouped df of predictions and thresholded test items per user
        - k: number of items for evaluation threshold
        
    Output: 
        - metrics (dict): key-value pairs for Precision@k, Recall@k, NDCG@k and MAP
    '''
    rdd = eval_df.select(
            "pred_items", "true_items"
        ).rdd.map(
            lambda row: (row.pred_items, row.true_items)
        )
    metrics = RM(rdd)
    
    return {
        f"precision_at_{k}": metrics.precisionAt(k),
        f"recall_at_{k}": metrics.recallAt(k),
        f"ndcg_at_{k}": metrics.ndcgAt(k),
        "map": metrics.meanAveragePrecision
    }