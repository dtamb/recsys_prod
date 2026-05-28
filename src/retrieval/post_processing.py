#post_processing.py

# Post processing on retrieval results

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import configs.settings as cfg

def remove_duplicates(df_rec, df_seen, k):
    '''
    Args: df_rec (data frame with recommendations)
        
        df_rec format:
            root
             |-- userId: integer (nullable = false)
             |-- recommendations: array (nullable = true)
             |    |-- element: struct (containsNull = true)
             |    |    |-- movieId: integer (nullable = true)
             |    |    |-- rating: float (nullable = true)
         
         df_seen (df with duplicates, i.e. training data)
     
    Output: filtered (df with items seen by users, i.e. training data)
    
        filtered format:
            root
             |-- userId: integer (nullable = true)
             |-- movieId: integer (nullable = true)
             |-- rating: float (nullable = true)
             |-- timestamp: timestamp (nullable = true)
        
    '''
    
    # Create df of seen films and group into sets
    seen = df_seen.select(cfg.USER_COL, cfg.ITEM_COL).distinct()

    seen_grouped = seen.groupBy(cfg.USER_COL).agg(
        F.collect_set(cfg.ITEM_COL).alias("seen_items")
    )
    
    # Create data frame with both seen films and recommendation films
    df_with_seen = df_rec.join(
        F.broadcast(seen_grouped),
        on=cfg.USER_COL,
        how="left"
    )
    
    # Filtering out films already seen
    filtered_arrays = df_with_seen.select(
        cfg.USER_COL,
        F.expr("""
            filter(recommendations, r -> NOT array_contains(seen_items, r.movieId))
        """).alias("filtered_recs")
    )
    
   # Expand out recommendations array and structure
    exploded = filtered_arrays.select(
        cfg.USER_COL,
        F.explode("filtered_recs").alias("rec")
    ).select(
        cfg.USER_COL,
        "rec.*"
    )
    
    
    # Create window function to  re-rank and select top k values
    window = Window.partitionBy(cfg.USER_COL).orderBy(F.desc(cfg.RATING_COL))
    filtered = exploded.withColumn(
        "rank", F.row_number().over(window)
    ).filter(
        F.col("rank") <= k
    ).drop("rank")
    
    return filtered

