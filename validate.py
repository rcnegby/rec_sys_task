import argparse
import logging
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from models import PopularRecommender, ALS
from utils import calc_MAP, get_coo_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-random_state', type=int, default=42)
    parser.add_argument('-topN', type=int, default=10)
    parser.add_argument('-interactions_path', type=str, default='data/interactions.csv')
    parser.add_argument('-test_size', type=float, default=0.2)
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, filename="validate.log", filemode="w")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("START validate.py")

    # read data
    logging.info("start read data")
    interactions = pd.read_csv(args.interactions_path)
    if 'row' not in interactions.columns or 'col' not in interactions.columns:
        logging.error("interactions must be DataFrame with columns row (it's user_id) and col (its item_id)")
        raise KeyError
    interactions = interactions.rename(columns={'row': 'user_id', 'col': 'item_id'}).drop(columns=['data'])
    logging.info("success read data")

    users_inv_mapping = dict(enumerate(interactions['user_id'].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    items_inv_mapping = dict(enumerate(interactions['item_id'].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}

    train_df, test_df = train_test_split(interactions, test_size=args.test_size, random_state=args.random_state)

    # get recommendations by PopularRecommender
    baseline = PopularRecommender()
    baseline.fit(train_df)
    logging.info("success fit PopularRecommender")
    base_rec = baseline.recommend(test_df.user_id.unique(), args.topN)
    logging.info("success recommend PopularRecommender")
    base_MAP = calc_MAP(test_df, base_rec)

    # get recommendations by AlternatingLeastSquares
    train_matrix = get_coo_matrix(train_df, users_mapping, items_mapping)
    als = ALS(users_mapping, items_inv_mapping)
    logging.info("start fit AlternatingLeastSquares")
    als.fit(train_matrix)
    logging.info("success fit AlternatingLeastSquares")
    logging.info("start recommend AlternatingLeastSquares")
    als_rec = als.recommend(test_df, train_matrix, args.topN)
    logging.info("success recommend AlternatingLeastSquares")
    als_MAP = calc_MAP(test_df, als_rec)

    print(f'MAP@{args.topN}:\n  PopularRecommender: {base_MAP}\n  AlternatingLeastSquares: {als_MAP}')
    logging.info("END validate.py")
