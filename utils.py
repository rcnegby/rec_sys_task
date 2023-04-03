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
