#tag_similarity.py

import configs.settings as cfg
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

def compute_tag_similarity(user_tag_norm, item_tag_norm, retrieval_df):
    
    # Join user tag scores to item tag scores for retrieval items
    similarity_score = retrieval_df.join(
        user_tag_norm, on=cfg.USER_COL, how='inner'
    ).join(
        item_tag_norm, on=cfg.ITEM_COL, how='inner'
    )
    
    # Convert to arrays and then dot product while still using Spark architecture
    # array items are pairwise multiplied with zip and then summed with aggregate
    similarity_score = similarity_score.withColumn(
        'user_arr', vector_to_array('user_tag_norm')
    ).withColumn(
        'item_arr', vector_to_array('item_tag_norm')
    ).withColumn(
        'similarity',
        F.aggregate(
            F.zip_with(
                'user_arr',
                'item_arr',
                lambda x, y: x * y
            ),
            F.lit(0.0),
            lambda acc, x: acc + x
        )
    )

    return similarity_score.select(cfg.USER_COL, cfg.ITEM_COL, F.col('rating').alias('als_score'), 'similarity')