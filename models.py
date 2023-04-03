import pandas as pd

from itertools import product
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


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
        '''
        create top of max_k most popular items
        '''

        self.top_recommend = interactions_df[self.item_col].value_counts().head(self.max_k).index.values

    def recommend(self, users: list = None, N: int = 10) -> pd.DataFrame:
        '''
        if user not None
        return DataFrame with top N most popular items for each user
        else return list of top N items
        '''

        rec = self.top_recommend[:N]
        if users is None:
            return rec
        else:
            combinations = list(product(users, rec))
            return pd.DataFrame({'user_id': [el[0] for el in combinations],
                                 'item_id': [el[1] for el in combinations],
                                 'rank': list(range(1, N+1)) * len(users)})


class ALS(AlternatingLeastSquares):
    '''
    wrapper over AlternatingLeastSquares for more convenient prediction
    '''

    def __init__(self, users_mapping: dict, items_inv_mapping: dict, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.users_mapping = users_mapping
        self.items_inv_mapping = items_inv_mapping
        self.mapper = None

    def fit(self, train_mat):
        '''
        use fit from parent class
        '''
        super().fit(train_mat)

    def generate_implicit_recs_mapper(self, recommend: callable, train_mat: csr_matrix, N: int) -> callable:
        '''
        function to predict for all users
        use function recommend from parent class
        '''

        def _recs_mapper(user):
            user_id = self.users_mapping[user]
            recs = recommend(user_id,
                             train_mat[user_id],
                             N=N)
            return [self.items_inv_mapping[item] for item in recs[0]]
        return _recs_mapper

    def recommend(self, test_df: pd.DataFrame, train_mat: csr_matrix,
                  N: int = 10, user_col: str = 'user_id') -> pd.DataFrame:
        '''
        return DataFrame with top N items for each user in test_df
        '''
        
        self.mapper = self.generate_implicit_recs_mapper(super().recommend, train_mat, N)
        recs = pd.DataFrame({
            user_col: test_df[user_col].unique()
        })
        recs['item_id'] = recs[user_col].map(self.mapper)
        recs = recs.explode('item_id')
        recs['rank'] = recs.groupby(user_col).cumcount() + 1
        return recs
