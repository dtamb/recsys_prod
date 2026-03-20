# prepocessing.py

# aggregate function options for pivot table function
from pyspark.sql.functions import (
    first, sum, avg, min, max, count, countDistinct
)

from pyspark.ml.feature import VectorAssembler, StandardScaler

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


def vectorise(df, exclude_cols, output_col='features'):
    '''
    Converts rows in a DataFrame into a single vector. 
    Excluded columns determine non-vectorised rows.
    
    Args:
        df: Spark DataFrame
        exclude_cols: can be in one of two formats
            - (str) column to exclude, e.g. 'movieId'
            - (dict of str) columns to exclude, e.g. '['userId', 'movieId']'
        output_col: (str) name of new vector column
        
    Returns:
        Spark DataFrame with selected columns now as a single vector column
    
    '''
    # Ensure exclude_cols is a list
    if isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
        
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    assembler = VectorAssembler(
        inputCols = feature_cols,
        outputCol = output_col
    )
    
    return assembler.transform(df).select(*exclude_cols, output_col)

def standardiser(
    df, input_col='features',
    output_col ='scaled_features',
    scale_mean = True, scale_std = True
):
    '''
    Standardises selected column of a Spark DataFrame.
    Options to standardise around mean=0 or std_dv =1
    
    NB: If requiring multiple columns standardised, use vectorise first.
    
    Args:
        df: Spark DataFrame
        input_col: column to scale
        ouput_col: name of new column
        scale_mean: True if mean is 0
        scale_std: True if std is 0
        
    Returns:
        scaled_df: Spark DataFrame with input column now standardised
        scaler_model: scaler model fit to dataset
    
    '''
    
    scaler = StandardScaler(
        inputCol = input_col,
        outputCol = output_col,
        withMean = scale_mean,
        withStd = scale_std
    )
    
    scaler_model = scaler.fit(df)
    
    scaled_df = scaler_model.transform(df).drop(input_col)
    
    return scaled_df, scaler_model
        
        
        
        
        