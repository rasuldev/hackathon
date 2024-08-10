import pandas as pd


def prepare_raw_dataset():
    sessions = pd.read_csv("../data/sessions.csv")
    campaigns = pd.read_csv("../data/campaigns.csv")
    charities = pd.read_csv("../data/charities.csv")
    keywords = pd.read_csv("../data/keywords.csv")

    campaigns_drop_cols = [
        "id",
        "hashtag",
        "collected",
        "user_count",
        "status",
        "link_open_event_count",
        "finished_at",
        "finish_payment_id",
        "description",
    ]
    charities_drop_cols = ["id", "short_description", "name", "address"]

    keywords = (
        pd.get_dummies(
            keywords, drop_first=False, columns=["keyword_id"], prefix="keyword"
        )
        .drop_duplicates("campaign_id")
        .drop(columns=["text"])
    )

    return (
        sessions.merge(campaigns, how="left", left_on="campaign_id", right_on="id")
        .drop(columns=campaigns_drop_cols)
        .merge(keywords, how="left", on="campaign_id")
        .merge(charities, how="left", left_on="charity_id", right_on="id")
        .drop(columns=charities_drop_cols)
        .fillna(0)
    )


def split_data_by_shares(df, shares):
    assert sum(shares.values()) == 1, "Shares must sum up to 1"

    total_sessions = df["session_id"].nunique()
    train_sessions_count = int(total_sessions * shares["train"])
    val_sessions_count = int(total_sessions * shares["val"])

    # Split data by sessions count
    unique_sessions = df["session_id"].unique()
    train_sessions = unique_sessions[:train_sessions_count]
    val_sessions = unique_sessions[
        train_sessions_count : train_sessions_count + val_sessions_count
    ]
    test_sessions = unique_sessions[train_sessions_count + val_sessions_count :]

    train_df = df[df["session_id"].isin(train_sessions)]
    val_df = df[df["session_id"].isin(val_sessions)]
    test_df = df[df["session_id"].isin(test_sessions)]

    return train_df, val_df, test_df
