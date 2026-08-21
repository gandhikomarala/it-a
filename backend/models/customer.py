# Customer and Behavioral Sub-models.
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from backend.database.base import Base, BaseEntityMixin

class Customer(Base, BaseEntityMixin):
    __tablename__ = "customers"
    customer_id = Column(String(64), nullable=False, unique=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    phone = Column(String(50), nullable=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    region = Column(String(100), nullable=False, index=True)
    city = Column(String(100), nullable=False)
    occupation = Column(String(100), nullable=True)
    income = Column(Float, nullable=False, default=0.0)
    signup_date = Column(Date, nullable=False)
    subscription_type = Column(String(50), nullable=False, index=True)
    contract_type = Column(String(50), nullable=False, index=True)
    payment_method = Column(String(50), nullable=False)
    monthly_charge = Column(Float, nullable=False, default=0.0)
    tenure_months = Column(Integer, nullable=False, default=0)
    total_spend = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, default=True, index=True)
    
    latest_churn_probability = Column(Float, nullable=True, index=True)
    latest_risk_level = Column(String(20), nullable=True, index=True)
    latest_prediction_date = Column(DateTime(timezone=True), nullable=True)
    
    tags = relationship("CustomerTag", back_populates="customer", cascade="all, delete-orphan")
    notes = relationship("CustomerNote", back_populates="customer", cascade="all, delete-orphan")
    events = relationship("CustomerEvent", back_populates="customer", cascade="all, delete-orphan")

class CustomerProfile(Base, BaseEntityMixin):
    __tablename__ = "customer_profiles"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, unique=True)
    company_name = Column(String(150), nullable=True)
    industry = Column(String(100), nullable=True)
    employee_count = Column(Integer, nullable=True)
    custom_metadata = Column(JSON, default=dict)

class CustomerUsage(Base, BaseEntityMixin):
    __tablename__ = "customer_usage"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    daily_usage_hours = Column(Float, default=0.0)
    weekly_usage_hours = Column(Float, default=0.0)
    monthly_usage_hours = Column(Float, default=0.0)
    login_count_monthly = Column(Integer, default=0)
    session_count_monthly = Column(Integer, default=0)
    average_session_duration_minutes = Column(Float, default=0.0)
    feature_usage_diversity_score = Column(Float, default=0.0)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    recorded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class CustomerPayment(Base, BaseEntityMixin):
    __tablename__ = "customer_payments"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    payment_amount = Column(Float, nullable=False)
    payment_status = Column(String(50), nullable=False)
    payment_failures_count = Column(Integer, default=0)
    late_payments_count = Column(Integer, default=0)
    refunds_count = Column(Integer, default=0)
    last_payment_date = Column(DateTime(timezone=True), nullable=True)

class CustomerSupport(Base, BaseEntityMixin):
    __tablename__ = "customer_support"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    ticket_count = Column(Integer, default=0)
    complaint_count = Column(Integer, default=0)
    average_resolution_hours = Column(Float, default=0.0)
    satisfaction_score = Column(Float, default=3.5)
    open_tickets_count = Column(Integer, default=0)
    last_ticket_date = Column(DateTime(timezone=True), nullable=True)

class CustomerSubscription(Base, BaseEntityMixin):
    __tablename__ = "customer_subscriptions"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    subscription_type = Column(String(50), nullable=False)
    contract_type = Column(String(50), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)
    auto_renew = Column(Boolean, default=True)
    plan_tier_upgrades = Column(Integer, default=0)
    plan_tier_downgrades = Column(Integer, default=0)
    cancellation_requested = Column(Boolean, default=False)

class CustomerEvent(Base, BaseEntityMixin):
    __tablename__ = "customer_events"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    event_metadata = Column(JSON, default=dict)
    customer = relationship("Customer", back_populates="events")

class CustomerNote(Base, BaseEntityMixin):
    __tablename__ = "customer_notes"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id = Column(String(36), nullable=False)
    note_text = Column(Text, nullable=False)
    customer = relationship("Customer", back_populates="notes")

class CustomerTag(Base, BaseEntityMixin):
    __tablename__ = "customer_tags"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    tag_name = Column(String(50), nullable=False, index=True)
    customer = relationship("Customer", back_populates="tags")
