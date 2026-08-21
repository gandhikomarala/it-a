#!/usr/bin/env python3
# Complete platform database seeder.
import asyncio
from datetime import datetime, timezone, date
from backend.database.session import async_session_factory, init_db
from backend.models.user import User, Organization
from backend.models.customer import Customer
from packages.utilities.security import PasswordHasher

async def seed():
    print("Initializing schema...")
    await init_db()

    async with async_session_factory() as db:
        print("Seeding demo organization and super-admin...")
        org = Organization(name="Enterprise Global Corp", slug="enterprise-global")
        db.add(org)
        await db.flush()

        admin_pw = PasswordHasher.hash_password("adminpassword123")
        admin = User(
            email="admin@enterprise-mlops.io",
            hashed_password=admin_pw,
            first_name="Platform",
            last_name="Administrator",
            role_name="SUPER_ADMIN",
            organization_id=org.id,
            is_active=True,
            is_verified=True
        )
        db.add(admin)

        print("Seeding sample customer cohort...")
        for i in range(1, 101):
            c = Customer(
                customer_id=f"CUS-{100000 + i}",
                first_name=f"Customer_{i}",
                last_name=f"Surname_{i}",
                email=f"customer.{i}@enterprise-client.io",
                age=35 + (i % 30),
                gender="Female" if i % 2 == 0 else "Male",
                region="North America" if i % 3 == 0 else "Europe",
                city="San Francisco" if i % 3 == 0 else "London",
                income=85000.0 + (i * 500),
                signup_date=date(2024, 1, 15),
                subscription_type="Premium" if i % 4 == 0 else "Standard",
                contract_type="Month-to-Month" if i % 2 == 0 else "One-Year",
                payment_method="Credit Card",
                monthly_charge=99.0 if i % 4 == 0 else 79.0,
                tenure_months=12 + (i % 24),
                total_spend=1200.0 + (i * 80),
                is_active=True,
                latest_churn_probability=0.82 if i % 5 == 0 else 0.15,
                latest_risk_level="HIGH" if i % 5 == 0 else "LOW",
                latest_prediction_date=datetime.now(timezone.utc)
            )
            db.add(c)

        await db.commit()
        print("Seeding completed successfully!")

if __name__ == '__main__':
    asyncio.run(seed())
