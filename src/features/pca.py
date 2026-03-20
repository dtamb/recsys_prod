# pca.py

from pyspark.ml.feature import PCA
import numpy as np
import matplotlib.pyplot as plt

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
        pca_df: Spark DataFrame with a vector column containing k-dimension
            features
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
        
    
def plot_pca_cumsum(cumulative_variance, threshold=0.9, plot_save_path=''):
    '''
    Plots the cumulative variances and a threshold line for PCA. 
    Option to save graph.
    
    Args: 
        cumulative_variance: explained variances over k for pca model (output from
            compute_pca_cumsum)
        threshold: percentage threshold goal for PCA
        plot_save_path: path to save plot (if empty, will not save)
        
    Returns:
        None
    '''
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.axhline(
        y=threshold, color='r', linestyle='--',
        label=f'{threshold*100:.0f}%threshold'
    )
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    
    # Saves plot if given file path 
    if plot_save_path:
        plt.savefig(plot_save_path, dpi=300)
    plt.show()