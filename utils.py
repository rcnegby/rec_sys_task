import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def get_coo_matrix(df: pd.DataFrame, users_mapping: dict, items_mapping: dict,
                   user_col: str = 'user_id', item_col: str = 'item_id'):
    '''
    transfotm pandas df to sparse matrix
    '''

    interaction_matrix = coo_matrix((
        np.ones(len(df), dtype=np.float32),
        (
            df[user_col].map(users_mapping.get),
            df[item_col].map(items_mapping.get)
        )
    ))
    return interaction_matrix.tocsr()


def calc_MAP(test_df: pd.DataFrame, recs: pd.DataFrame):
    '''
    calculate MAP@K
    test_df should contain columns ['user_id', 'item_id']
    recs should contain columns ['user_id', 'item_id', 'rank']
    test_df stores interactions
    recs stores recommended objects with rank for each user
    '''

    test_recs = test_df.merge(recs, on=['user_id', 'item_id'], how='left')
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])

    test_recs['users_item_count'] = test_recs.groupby(by='user_id')['rank'].transform(np.size)
    test_recs['precision@K'] = test_recs.groupby(by='user_id').cumcount() + 1
    test_recs['precision@K'] = test_recs['precision@K'] / test_recs['rank']

    users_count = test_recs.user_id.nunique()
    return (test_recs['precision@K'] / test_recs['users_item_count']).sum() / users_count
