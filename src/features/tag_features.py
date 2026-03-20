# item_features.py

# Import generic pivot function
from data.preprocessing import pivot_table

# Import column names
import configs.settings as cfg

# aggregate function options for pivot table function
from pyspark.sql.functions import (
    first, sum, avg, min, max, count, countDistinct
)

def pivot_genome_scores (
    df,
    index_col=cfg.ITEM_COL, # movieId
    column_col=cfg.TAG_ID_COL, # tagId
    value_col=cfg.RELEVANCE_COL, # relevance
    agg_func=first, # default required by Spark
    fill_value=0 # value to replace nulls
    ): 
    
    '''
    Pivot genome scores from long --> wide format and handle nulls.
    
    Args:
        df: Spark DataFrame
        index_col: column to group by (becomes rows)
        column_col: column to pivot into mulitple columns
        value_col: values to fill in the pivot table
        agg_func: aggregation function from pyspark.sql.functions (default: first)
        fill_value: value to fill nulls
        
    Returns:
        Spark DataFrame in wide format, nulls filled  
    '''
    
    pivoted = pivot_table(df, index_col, column_col, value_col, agg_func)
    return pivoted.fillna(fill_value).orderBy(index_col)
    