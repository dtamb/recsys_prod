# prepocessing.py

# aggregate function options for pivot table function
from pyspark.sql.functions import (
    first, sum, avg, min, max, count, countDistinct
)

def pivot_table(df, index_col, column_col, value_col, agg_func=first):
    '''
    Pivots a Spark DataFrame from long --> wide format.
    
    Args: 
        df: Spark DataFrame
        index_col: column to group by (becomes rows)
        column_col: column to pivot into mulitple columns
        value_col: values to fill in the pivot table
        agg_func: aggregation function from pyspark.sql.functions (default: first)
        
    Returns:
        Spark DataFrame in wide format
    
    '''
    return df.groupBy(index_col).pivot(column_col).agg(agg_func(value_col))