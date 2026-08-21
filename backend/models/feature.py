# Feature models.
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text, JSON
from backend.database.base import Base, BaseEntityMixin

class FeatureDefinition(Base, BaseEntityMixin):
    __tablename__ = "features"
    name = Column(String(100), nullable=False, unique=True, index=True)
    data_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    transformation_logic = Column(Text, nullable=True)
    category = Column(String(50), nullable=False, index=True)
    is_target = Column(Boolean, default=False)

class FeatureVersion(Base, BaseEntityMixin):
    __tablename__ = "feature_versions"
    feature_id = Column(String(36), ForeignKey("features.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    formula = Column(Text, nullable=True)

class FeatureSet(Base, BaseEntityMixin):
    __tablename__ = "feature_sets"
    name = Column(String(150), nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(Integer, default=1)
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    feature_names = Column(JSON, default=list)
    target_column = Column(String(50), default="churn")

class FeatureValue(Base, BaseEntityMixin):
    __tablename__ = "feature_values"
    customer_id = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    feature_value = Column(Float, nullable=True)
