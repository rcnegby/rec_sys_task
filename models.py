import pandas as pd
from itertools import product


class PopularRecommender():
    '''
    baseline recommender, that recommend N
    most popular items
    '''

    def __init__(self, item_col: str = 'item_id', max_k: int = 100):
        self.item_col = item_col
        self.max_k = max_k
        self.top_recommend = []

    def fit(self, interactions_df: pd.DataFrame):
        self.top_recommend = interactions_df[self.item_col].value_counts().head(self.max_k).index.values

    def recommend(self, users: list = None, N: int = 10):
        rec = self.top_recommend[:N]
        if users is None:
            return rec
        else:
            combinations = list(product(users, rec))
            return pd.DataFrame({'user_id': [el[0] for el in combinations],
                                 'item_id': [el[1] for el in combinations],
                                 'rank': list(range(1, N+1)) * len(users)})
