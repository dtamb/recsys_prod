# als_trainer.py


from pyspark.ml.recommendation import ALS
def train_als(train_df, config):
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