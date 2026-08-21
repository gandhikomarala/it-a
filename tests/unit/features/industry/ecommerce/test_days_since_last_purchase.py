# Unit Test for DaysSinceLastPurchase (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.days_since_last_purchase import DaysSinceLastPurchase
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_days_since_last_purchase_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DaysSinceLastPurchase()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"days_since_last_purchase_signal" in res.columns
    assert f"days_since_last_purchase_risk_index" in res.columns
    assert not res[f"days_since_last_purchase_signal"].isnull().any()

def test_days_since_last_purchase_empty_handling():
    extractor = DaysSinceLastPurchase()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
