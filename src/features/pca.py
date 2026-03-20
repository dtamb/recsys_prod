# pca.py

from pyspark.ml.feature import PCA
import numpy as np

def fit_pca(df_scaled, item_col, input_col="scaled_features", output_col="pca_features", k=100):
    
    '''
    Simplifies a scaled and vectorised dataset to k-dimensions using PCA.
    
    Args: 
        df_scaled: Spark DataFrame with a scaled and vectorised column to reduce
            dimensions of
        item_col: items columns
        input_col: scaled and vectorised column of features
        output_col: reduced dimensions column name
        k (int): number of feature dimensions
    
    Returns (list):
        pca_df: Spark DataFrame with a vector column containing k-dimension features
        pca_model: Fitted PCA Model
    '''
    
    pca = PCA(k=k, inputCol=input_col, outputCol=output_col)
    pca_model = pca.fit(df_scaled)
    
    pca_df = pca_model.transform(df_scaled).select(item_col, output_col)
    
    return [pca_df, pca_model]

def compute_pca_cumsum(pca_model):
    '''
    Takes a PCA model and returns the cumulative sum of variances.
    
    Args:
        Spark PCA model
        
    Returns:
        Cumulative Variance Scores over k
        
    '''
    
    explained_var = np.array(pca_model.explainedVariance)
    cumulative = explained_var.cumsum()
    
    return cumulative
        
    
    