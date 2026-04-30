# user_vectors.py

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array, array_to_vector
import configs.settings as cfg
from pyspark.ml.stat import Summarizer

def build_user_tags(weights_df, tags_df):
    
    '''
    Creates user tags that are representative of their film choices separate from popular films.
    
    user_tag = sum(weight_i * tag_i) / sum(abs(weight_i))
    
    NB: dividing by absolute weights to preserve both magnitude and polarity of weightings
    
    Args:
        weights_df: Spark DF from features/biases.py compute_user_weights function.
            contains userID, movieID, weight
        tags_df: Spark DF from features/tag_features.py build_genome_pca_features
            contains movieId and pca_features
               
    Returns:
        user_vector: Spark DF with userID and user_tag (vector of user tags)
        
    '''
    
    # Create column of vectors of weights * film_tags
    user_vector = weights_df.join(
        tags_df, on=cfg.ITEM_COL, how='inner'
    ).withColumn(
            'weighted_tags',
            array_to_vector(
                F.transform(
                    vector_to_array(F.col('pca_features')),
                    lambda x: x * F.col('weight')
                )
            )
    )

    # Compute user tags by summing over weights * film_tags and dividing by
    # the sum of absolute weight values
    user_vector = user_vector.groupBy(cfg.USER_COL).agg(
        Summarizer.sum(F.col('weighted_tags')).alias('sum_user_tags'),
        F.sum(F.abs('weight')).alias('sum_abs_weight')
    ).withColumn(
        'user_tag',
        array_to_vector(
            F.transform(
                vector_to_array(F.col('sum_user_tags')),
                lambda x: F.when(F.col('sum_abs_weight') != 0, x / F.col('sum_abs_weight')).otherwise(0.0)
            )
        )
    )
    
    return user_vector.select(cfg.USER_COL, 'user_tag')