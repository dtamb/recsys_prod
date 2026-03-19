# als_trainer.py
from .post_processing import remove_duplicates
from pyspark.ml.recommendation import ALS

def generate_candidates(model, train, k):
    '''
        Two-steps process for generating candidates.
        1) Creating standard rec function using the model's method
        2) Calling filter function to remove any recommended items that users have
        already seen. (Although ALS' recommendForAllUsers removes already seen
        films from the recommendations dataframe, some seen films do remain.)

        Agrs:
            - model: model generated from ALS training
            - train: training data for duplicate comparison
            - k: number of candidates (due to filtering, actual candidate number
            will be <=k)
        Returns:
            List of recommendations per user, filtered for already seen items
    '''
    recs = model.recommendForAllUsers(k)
    
    return remove_duplicates(recs, train)


def train(train_df, config):
    '''
    Args: train_df, config
        train_df: userId, itemId, rating
        config: (dict) of parameters
        E.g.
            config = {
            'alpha': 20,
            'maxIter': 15,
            'rank': 50,
            'implicitPrefs': True,
            'regParam': 0.05,
            'coldStartStrategy':'drop',
            'nonnegative': False,
            'seed': 2026,
            'userCol': USER_COL,
            'itemCol': ITEM_COL,
            'ratingCol':RATING_COL
            }

    Returns: model
        in form: als.fit(train_df)
    '''

    als = ALS(
        alpha=config['alpha'],
        maxIter=config['maxIter'],
        rank=config['rank'],
        implicitPrefs=config['implicitPrefs'],
        regParam=config['regParam'],
        coldStartStrategy=config['coldStartStrategy'],
        nonnegative=config['nonnegative'],
        seed=config['seed'],
        userCol=config['userCol'],
        itemCol=config['itemCol'],
        ratingCol=config['ratingCol']
    )
    
    return als.fit(train_df)