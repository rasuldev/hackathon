import pandas as pd


class SessionTable:
    def __init__(self, campaigns, payments):
        self.campaigns = campaigns
        self.payments = payments

    def generate_for_user(self, user_id):
        start_date = self.campaigns["published_at"].min().floor('d')
        end_date = self.payments["finished_at"].max().floor('d')
        session_table = []
        for delta in range((end_date - start_date).days):
            date = start_date + pd.DateOffset(delta)
            campaigns_feed = self.get_campaigns_feed(date, user_id)
            for pos in range(len(campaigns_feed)):
                campaign_id = campaigns_feed.iloc[pos].id
                session_table.append(
                    {'date': date, 'user_id': user_id, 'campaign_id': campaign_id, 'pos': pos,
                     'is_donated': self.is_donated(date, campaign_id, user_id)})
        return pd.DataFrame(session_table)

    def get_campaigns_feed(self, date, user_id):
        df = self.campaigns
        # keep only active campaigns at the given date
        df = df[(df.published_at <= date) & ((df.finished_at >= date) | (df.finished_at.isnull()))]

        # campaigns from favourite charities should be higher: will be implemented later using user_id

        # recently published campaigns should be higher
        return df.sort_values(['published_at']).head(10)

    def is_donated(self, date, campaign_id, user_id):
        payments = self.payments
        return not payments[(payments.finished_at.dt.floor('d') == date.floor('d')) & (payments.user_id == user_id) & (
                payments.campaign_id == campaign_id)].empty
