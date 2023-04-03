import argparse
import logging
import sys
import pandas as pd

from models import PopularRecommender, ALS
from utils import get_coo_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-random_state', type=int, default=42)
    parser.add_argument('-topN', type=int, default=10)
    parser.add_argument('-interactions_path', type=str, default='data/interactions.csv')
    parser.add_argument('-user_region_path', type=str, default='data/user_region.csv')
    parser.add_argument('-user_age_path', type=str, default='data/user_age.csv')
    parser.add_argument('-cold_recommend_save_path', type=str, default='data/cold_users_recommend.csv')
    parser.add_argument('-recommend_save_path', type=str, default='data/users_recommend.csv')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, filename="eval.log", filemode="w")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("START eval.py")

    # read data
    logging.info("start read data")
    interactions = pd.read_csv(args.interactions_path)
    if 'row' not in interactions.columns or 'col' not in interactions.columns:
        logging.error("interactions must be DataFrame with columns row (it's user_id) and col (its item_id)")
        raise KeyError
    interactions = interactions.rename(columns={'row': 'user_id', 'col': 'item_id'}).drop(columns=['data'])

    # find cold users
    user_region_users = set(pd.read_csv(args.user_region_path)['row'])
    user_age_users = set(pd.read_csv(args.user_age_path)['row'])
    interactions_users = set(interactions['user_id'])
    cold_users = list((user_age_users | user_region_users) - interactions_users)

    logging.info("success read data")

    # some auxiliary dictionaries for sparse matrix
    users_inv_mapping = dict(enumerate(interactions['user_id'].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    items_inv_mapping = dict(enumerate(interactions['item_id'].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}

    # get recommendations by PopularRecommender for cold users
    baseline = PopularRecommender()
    baseline.fit(interactions)
    logging.info("success fit PopularRecommender")
    base_rec = baseline.recommend(cold_users, args.topN)
    logging.info("success recommend PopularRecommender")
    base_rec.to_csv(args.cold_recommend_save_path, index=False)
    logging.info(f"success save cold users at {args.cold_recommend_save_path}")

    # get recommendations by AlternatingLeastSquares
    interactions_matrix = get_coo_matrix(interactions, users_mapping, items_mapping)
    als = ALS(users_mapping, items_inv_mapping)
    logging.info("start fit AlternatingLeastSquares")
    als.fit(interactions_matrix)
    logging.info("success fit AlternatingLeastSquares")
    logging.info("start recommend AlternatingLeastSquares")
    als_rec = als.recommend(interactions, interactions_matrix, args.topN)
    logging.info("success recommend AlternatingLeastSquares")
    als_rec.to_csv(args.recommend_save_path, index=False)
    logging.info(f"success save users at {args.recommend_save_path}")

    logging.info("END eval.py")
