#post_processing.py

# Post processing on retrieval results

from pyspark.sql import functions as F

def remove_duplicates(df_rec, df_seen):
    '''
    Args: df_rec (data frame with recommendations)
        
        df_rec format:
            root
             |-- userId: integer (nullable = false)
             |-- recommendations: array (nullable = true)
             |    |-- element: struct (containsNull = true)
             |    |    |-- movieId: integer (nullable = true)
             |    |    |-- rating: float (nullable = true)
         
         train (df with dupes, i.e. training data)
     
    Output: df_seen (df with items seen by users, i.e. training data)
    
        df_seen format:
            root
             |-- userId: integer (nullable = true)
             |-- movieId: integer (nullable = true)
             |-- rating: float (nullable = true)
             |-- timestamp: timestamp (nullable = true)
        
    '''
    exploded = df_rec.select(
    df_rec.userId,
    F.explode('recommendations').alias('rec')
    ).select(
        'userId',
        'rec.*'
    )

    filtered = exploded.join(
        df_seen.select('userId', 'movieId'),
        on=['userId', 'movieId'],
        how='left_anti'
    ).orderBy(
        ['userId', 'rating'],
        ascending=[True, False]
    )
    
    return filtered