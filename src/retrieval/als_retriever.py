# als_retriever.py

'''
Two-steps process for generating candidates.
1) Creating standard rec function using the model's method
2) Calling filter function to remove any recommended items that users have already seen. (Although ALS' recommendForAllUsers removes already seen films from the recommendations dataframe, some seen films do remain.)

Agrs:
    - model: model generated from ALS training
    - train: training data for duplicate comparison
    - k: number of candidates (due to filtering, actual candidate number will be <=k)
'''

from .post_processing import remove_duplicates

def generate_als_candidates(model, train, k):
    recs = model.recommendForAllUsers(k)
    
    return remove_duplicates(recs, train)