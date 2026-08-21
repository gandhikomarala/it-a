# Comprehensive Unit Test for CommittedSpendBurndownVelocityExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.committed_spend_burndown_velocity import CommittedSpendBurndownVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_committed_spend_burndown_velocity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CommittedSpendBurndownVelocityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"committed_spend_burndown_velocity_signal" in res.columns
    assert f"committed_spend_burndown_velocity_risk_score" in res.columns
    assert not res[f"committed_spend_burndown_velocity_signal"].isnull().any()

def test_committed_spend_burndown_velocity_empty_handling():
    extractor = CommittedSpendBurndownVelocityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
