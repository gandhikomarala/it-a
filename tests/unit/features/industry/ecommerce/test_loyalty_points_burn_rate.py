# Unit Test for LoyaltyPointsBurnRate (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.loyalty_points_burn_rate import LoyaltyPointsBurnRate
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_loyalty_points_burn_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LoyaltyPointsBurnRate()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"loyalty_points_burn_rate_signal" in res.columns
    assert f"loyalty_points_burn_rate_risk_index" in res.columns
    assert not res[f"loyalty_points_burn_rate_signal"].isnull().any()

def test_loyalty_points_burn_rate_empty_handling():
    extractor = LoyaltyPointsBurnRate()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
