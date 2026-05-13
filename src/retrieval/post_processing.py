#post_processing.py

# Post processing on retrieval results

from pyspark.sql import functions as F
from pyspark.sql.window import Window

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
    
    # Expand out recommendations array and structure
    exploded = df_rec.select(
        df_rec.userId,
        F.explode('recommendations').alias('rec')
    ).select(
        'userId',
        'rec.*'
    )

    # Remove recommendations that are already seen
    filtered = exploded.join(
        df_seen.select('userId', 'movieId'),
        on=['userId', 'movieId'],
        how='left_anti'
    ).orderBy(
        ['userId', 'rating'],
        ascending=[True, False]
    )
    
    # Create window function to  re-rank and select top k values
    window = Window.partitionBy("userId").orderBy(F.desc("rating"))

    filtered = filtered.withColumn(
        "rank", F.row_number().over(window)
    ).filter(
        F.col("rank") <= k
    ).drop("rank")
    
    return filtered

