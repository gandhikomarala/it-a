# Comprehensive Unit Test for OrderFrequencyDecaySlopeExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.order_frequency_decay_slope import OrderFrequencyDecaySlopeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_order_frequency_decay_slope_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OrderFrequencyDecaySlopeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"order_frequency_decay_slope_signal" in res.columns
    assert f"order_frequency_decay_slope_risk_score" in res.columns
    assert not res[f"order_frequency_decay_slope_signal"].isnull().any()

def test_order_frequency_decay_slope_empty_handling():
    extractor = OrderFrequencyDecaySlopeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
