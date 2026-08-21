# Comprehensive Unit Test for AverageBasketSizeDriftExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.average_basket_size_drift import AverageBasketSizeDriftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_average_basket_size_drift_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AverageBasketSizeDriftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"average_basket_size_drift_signal" in res.columns
    assert f"average_basket_size_drift_risk_score" in res.columns
    assert not res[f"average_basket_size_drift_signal"].isnull().any()

def test_average_basket_size_drift_empty_handling():
    extractor = AverageBasketSizeDriftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
