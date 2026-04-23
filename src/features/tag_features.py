# tag_features.py

from data.preprocessing import pivot_table, vectorise, standardiser
from features.pca import fit_pca
import configs.settings as cfg

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
    return pivoted.fillna(fill_value)
    
def build_genome_pca_features(genome_df, k=45, scale_std=False):
    
    '''
    Creates scaler model + pca model and df for genome dataset.
    
    Args:
        genome_df: Spark DataFrame
        k: reduced dimensions for PCA
        scale_std: flag for scaling standard deviation to 1
        
    Returns:
        genome_scaler_model: scaler model trained on genome data
        genome_pca_df: Spark DataFrame of reduced dimensions genome data
        genome_pca_model: PCA model for reducing genome dimensions
        
    '''
    
    # Pivot and vectorise genome data (using default values)
    vector_df = vectorise(pivot_genome_scores(genome_df), cfg.ITEM_COL)
    
    # Standardise and create scaler model
    genome_std, genome_scaler_model = standardiser(vector_df, scale_std=scale_std)
    
    # Run PCA on genome dataset to create PCA dataset & model
    genome_pca_df, genome_pca_model = fit_pca(genome_std, item_col=cfg.ITEM_COL,
                                              k=k)
    
    return genome_scaler_model, genome_pca_df, genome_pca_model
    
    

    
    