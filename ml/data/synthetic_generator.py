# Enterprise synthetic customer data generator with non-linear churn dynamics.
import random
from datetime import date, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class SyntheticCustomerGenerator:
    # Generates realistic customer cohorts with deep behavioral correlations.

    REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"]
    CITIES = {
        "North America": ["New York", "San Francisco", "Austin", "Toronto", "Chicago", "Seattle"],
        "Europe": ["London", "Berlin", "Paris", "Amsterdam", "Stockholm", "Dublin"],
        "Asia-Pacific": ["Singapore", "Tokyo", "Sydney", "Bangalore", "Seoul"],
        "Latin America": ["São Paulo", "Mexico City", "Buenos Aires", "Bogotá"],
        "Middle East & Africa": ["Dubai", "Riyadh", "Tel Aviv", "Cape Town"]
    }
    OCCUPATIONS = [
        "Software Engineer", "Product Manager", "Data Analyst", "Marketing Director",
        "Financial Analyst", "Operations Manager", "Sales Representative", "HR Specialist",
        "Healthcare Worker", "Consultant", "Designer", "Executive", "Other"
    ]
    SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium", "Enterprise"]
    CONTRACT_TYPES = ["Month-to-Month", "One-Year", "Two-Year"]
    PAYMENT_METHODS = ["Credit Card", "Bank Transfer", "Electronic Check", "PayPal"]

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def generate(self, n_customers: int = 5000) -> pd.DataFrame:
        logger.info(f"Generating {n_customers:,} realistic synthetic customer records...")

        records: List[Dict[str, Any]] = []
        base_date = date.today()

        for i in range(1, n_customers + 1):
            customer_id = f"CUS-{100000 + i}"
            age = int(np.clip(np.random.normal(40, 12), 18, 85))
            gender = random.choice(["Male", "Female", "Non-Binary", "Other"])
            region = random.choice(self.REGIONS)
            city = random.choice(self.CITIES[region])
            occupation = random.choice(self.OCCUPATIONS)
            income = float(np.clip(np.random.lognormal(10.8, 0.6), 25000, 350000))

            tenure_months = int(np.clip(np.random.exponential(18), 1, 72))
            signup_date = base_date - timedelta(days=int(tenure_months * 30.4))

            subscription = np.random.choice(self.SUBSCRIPTION_TYPES, p=[0.35, 0.40, 0.18, 0.07])
            contract = np.random.choice(self.CONTRACT_TYPES, p=[0.55, 0.30, 0.15])
            payment_method = random.choice(self.PAYMENT_METHODS)

            price_map = {"Basic": 29.0, "Standard": 79.0, "Premium": 149.0, "Enterprise": 399.0}
            base_charge = price_map[subscription] + random.uniform(-5.0, 15.0)
            monthly_charge = round(float(np.clip(base_charge, 19.0, 600.0)), 2)
            total_spend = round(monthly_charge * tenure_months * random.uniform(0.95, 1.05), 2)

            avg_daily_usage = float(np.clip(np.random.normal(2.5, 1.2), 0.1, 8.0))
            monthly_usage = round(avg_daily_usage * 30.4, 1)
            login_count = int(np.clip(np.random.poisson(avg_daily_usage * 6), 1, 90))
            days_since_last_login = int(np.clip(np.random.exponential(4), 0, 45))

            payment_failures = int(np.random.choice([0, 1, 2, 3, 4], p=[0.75, 0.14, 0.07, 0.03, 0.01]))
            late_payments = int(np.random.choice([0, 1, 2, 3], p=[0.80, 0.12, 0.06, 0.02]))

            ticket_count = int(np.random.poisson(1.5))
            complaint_count = int(np.random.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.07, 0.03]))
            satisfaction_score = round(float(np.clip(np.random.normal(3.8, 0.9) - (complaint_count * 0.5), 1.0, 5.0)), 1)
            usage_trend = float(np.clip(np.random.normal(0.05, 0.35) - (days_since_last_login * 0.02), -1.0, 1.0))

            logit = -2.2
            if contract == "Month-to-Month":
                logit += 0.85
            elif contract == "Two-Year":
                logit -= 0.90

            logit += (payment_failures * 0.70)
            logit += (late_payments * 0.45)
            logit += (complaint_count * 0.65)
            logit -= ((satisfaction_score - 3.0) * 0.55)
            logit += (days_since_last_login * 0.08)
            logit -= (usage_trend * 1.20)
            logit -= (avg_daily_usage * 0.30)
            logit -= (np.log1p(tenure_months) * 0.40)

            if subscription == "Basic" and monthly_charge > 35.0:
                logit += 0.40

            churn_prob = 1.0 / (1.0 + np.exp(-logit))
            churn_prob = float(np.clip(churn_prob, 0.01, 0.99))
            churn = 1 if random.random() < churn_prob else 0

            records.append({
                "customer_id": customer_id,
                "first_name": f"Customer_{i}",
                "last_name": f"Surname_{i}",
                "email": f"customer.{i}@enterprise-client.io",
                "phone": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "age": age,
                "gender": gender,
                "region": region,
                "city": city,
                "occupation": occupation,
                "income": income,
                "signup_date": str(signup_date),
                "subscription_type": subscription,
                "contract_type": contract,
                "payment_method": payment_method,
                "monthly_charge": monthly_charge,
                "tenure_months": tenure_months,
                "total_spend": total_spend,
                "daily_usage_hours": avg_daily_usage,
                "monthly_usage_hours": monthly_usage,
                "login_count_monthly": login_count,
                "days_since_last_login": days_since_last_login,
                "payment_failures_count": payment_failures,
                "late_payments_count": late_payments,
                "ticket_count": ticket_count,
                "complaint_count": complaint_count,
                "satisfaction_score": satisfaction_score,
                "usage_trend": round(usage_trend, 3),
                "true_churn_probability": round(churn_prob, 4),
                "churn": churn
            })

        df = pd.DataFrame(records)
        churn_rate = df['churn'].mean() * 100.0
        logger.info(f"Generated {len(df):,} records with realistic churn rate: {churn_rate:.2f}%")
        return df
