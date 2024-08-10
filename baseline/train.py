import click
import pandas as pd
from enum import Enum
from catboost import Pool, CatBoostRanker
from dataset import prepare_raw_dataset, split_data_by_shares


TARGET = "donation_count"


class Features(Enum):
    CAMPAIGN_GOAL = "goal"
    CAMPAIGN_HELP_RECEIVER_COUNT = "help_receiver_count"
    CHARITY_SUBSCRIBERS_COUNT = "subscribers_count"


class CategoricalFeatures(Enum):
    CHARITY_ID = "charity_id"
    CHARITY_REGION_ID = "region_id"
    KEYWORD_2 = "keyword_2"
    KEYWORD_3 = "keyword_3"
    KEYWORD_4 = "keyword_4"
    KEYWORD_5 = "keyword_5"
    KEYWORD_6 = "keyword_6"
    KEYWORD_7 = "keyword_7"
    KEYWORD_8 = "keyword_8"
    KEYWORD_9 = "keyword_9"
    KEYWORD_10 = "keyword_10"
    KEYWORD_11 = "keyword_11"
    KEYWORD_12 = "keyword_12"
    KEYWORD_14 = "keyword_14"
    KEYWORD_15 = "keyword_15"
    KEYWORD_16 = "keyword_16"
    KEYWORD_17 = "keyword_17"
    KEYWORD_18 = "keyword_18"
    KEYWORD_19 = "keyword_19"
    KEYWORD_20 = "keyword_20"
    KEYWORD_21 = "keyword_21"
    KEYWORD_23 = "keyword_23"
    KEYWORD_24 = "keyword_24"
    KEYWORD_26 = "keyword_26"


def prepare_catboost_pools(sessions: pd.DataFrame, shares: dict):
    features = [e.value for e in Features]
    cat_features = [e.value for e in CategoricalFeatures]

    # Attention: all objects in dataset must be sorted by group_id (that is session_id in our case) and position
    # Read this for proper understanding: https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    df = sessions.sort_values(["session_id", "pos"])

    train_df, val_df, test_df = split_data_by_shares(df, shares)

    train_pool = Pool(
        data=train_df[features + cat_features],
        label=train_df[TARGET],
        group_id=train_df["session_id"],
        cat_features=cat_features,
    )

    val_pool = Pool(
        data=val_df[features + cat_features],
        label=val_df[TARGET],
        group_id=val_df["session_id"],
        cat_features=cat_features,
    )

    test_pool = Pool(
        data=test_df[features + cat_features],
        label=test_df[TARGET],
        group_id=test_df["session_id"],
        cat_features=cat_features,
    )

    return train_pool, val_pool, test_pool


@click.command()
@click.option("--name", default="my_catboost_ranker")
@click.option("--iterations", default=1000)
@click.option("--lr", default=0.1)
@click.option("--save_model_path", default=".")
def train(name: str, iterations: int, lr: float, save_model_path: str):
    sessions = prepare_raw_dataset()
    train_pool, val_pool, test_pool = prepare_catboost_pools(
        sessions, {"train": 0.7, "val": 0.15, "test": 0.15}
    )

    ranker = CatBoostRanker(
        loss_function="YetiRank",
        iterations=iterations,
        learning_rate=lr,
        depth=6,
        eval_metric="NDCG",
        random_seed=42,
        logging_level="Verbose",
    )

    ranker.fit(train_pool, eval_set=val_pool, use_best_model=True)

    test_ndcg = ranker.eval_metrics(test_pool, metrics=["NDCG"])["NDCG:type=Base"][-1]
    print(f"Test NDCG: {test_ndcg}")

    ranker.save_model(f"{save_model_path}/{name}.cbm")


if __name__ == "__main__":
    train()
