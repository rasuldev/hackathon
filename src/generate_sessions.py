import click
import hashlib
import pandas as pd
from datetime import timedelta
from tqdm.auto import tqdm


def rfind(lst, element):
    element_indices = [index for index, value in enumerate(lst) if value == element]
    return max(element_indices) if element_indices else None


def aggregate_payments(session_payments):
    """Aggregate payments for the same campaign_id within a session."""
    aggregated_payments = {}

    for payment in session_payments:
        campaign_id = payment["campaign_id"]
        if campaign_id not in aggregated_payments:
            aggregated_payments[campaign_id] = {
                "donation_count": 0,
                "amount": 0,
                "payment_ids": [],
            }

        aggregated_payments[campaign_id]["donation_count"] += 1
        aggregated_payments[campaign_id]["amount"] += payment["amount"]
        aggregated_payments[campaign_id]["payment_ids"].append(payment["id"])

    return aggregated_payments


def generate_sessions(
    payments: pd.DataFrame,
    campaigns: pd.DataFrame,
    session_count: int,
    max_position: int,
    merge_payments_within_seconds: int,
):
    payments["finished_at"] = pd.to_datetime(payments["finished_at"])
    campaigns["published_at"] = pd.to_datetime(campaigns["published_at"])
    campaigns["finished_at"] = pd.to_datetime(campaigns["finished_at"])

    payments = payments.sort_values(["user_id", "finished_at"])

    sessions = []

    # Merge payments to form sessions
    current_user = None
    current_session_start = None
    session_payments = []

    for _, row in tqdm(payments.iterrows()):
        if len(sessions) == session_count:
            break

        if current_user != row["user_id"] or (
            row["finished_at"] - current_session_start
            > timedelta(seconds=merge_payments_within_seconds)
        ):
            # If there's a switch in user or timeout between payments, create a new session
            if session_payments:
                session_hash = hashlib.md5(
                    (current_user + str(current_session_start)).encode()
                ).hexdigest()
                sessions.append(
                    (
                        session_hash,
                        current_user,
                        current_session_start,
                        session_payments,
                    )
                )
                session_payments = []
            current_user = row["user_id"]
            current_session_start = row["finished_at"]

        session_payments.append(row)

    result = []

    for session_id, user_id, session_ts, session_payments in tqdm(sessions):
        # Aggregate payments for the same campaign_id
        aggregated_payments = aggregate_payments(session_payments)

        # Find active campaigns at the time of session_ts
        active_campaigns = campaigns[
            (campaigns["published_at"] <= session_ts)
            & (campaigns["finished_at"] >= session_ts)
        ].copy()

        # Sort campaigns by published_at
        active_campaigns.sort_values("published_at", ascending=False, inplace=True)
        active_campaigns.reset_index(drop=True, inplace=True)

        # Add position column
        active_campaigns["pos"] = active_campaigns.index

        # Limit to max_position arg or max payment position
        max_payment_position = rfind(
            [
                campaign_id in aggregated_payments.keys()
                for campaign_id in list(active_campaigns.id)
            ],
            True,
        )
        limit_position = max(
            max_position, max_payment_position if max_payment_position else 0
        )

        active_campaigns = active_campaigns[active_campaigns["pos"] <= limit_position]

        for _, campaign_row in active_campaigns.iterrows():
            campaign_id = campaign_row["id"]
            if campaign_id in aggregated_payments:
                # Create the result row
                result_row = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "session_ts": session_ts,
                    "pos": campaign_row["pos"],
                    "campaign_id": campaign_row["id"],
                    "donation_count": aggregated_payments[campaign_id][
                        "donation_count"
                    ],
                    "amount": aggregated_payments[campaign_id]["amount"],
                    "payment_ids": aggregated_payments[campaign_id]["payment_ids"],
                }
                result.append(result_row)
            else:
                # Create the result row for non-donated campaigns
                result_row = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "session_ts": session_ts,
                    "pos": campaign_row["pos"],
                    "campaign_id": campaign_row["id"],
                    "donation_count": 0,
                    "amount": 0,
                    "payment_ids": [],
                }
                result.append(result_row)

    session_log_df = pd.DataFrame(result)
    return session_log_df


@click.command()
@click.option("--output", default="./data/sessions.csv")
@click.option("--payments", default="./data/payments.csv")
@click.option("--campaigns", default="./data/campaigns.csv")
@click.option("--session_count", default=30000)
@click.option("--max_position", default=10)
def main(
    output: str, payments: str, campaigns: str, session_count: int, max_position: int
):
    sessions = generate_sessions(
        payments=pd.read_csv(payments),
        campaigns=pd.read_csv(campaigns),
        session_count=session_count,
        max_position=max_position,
        merge_payments_within_seconds=600,
    )
    sessions.to_csv(output, index=False)


if __name__ == "__main__":
    main()
