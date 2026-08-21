# Comprehensive Unit Test for OrderInaccuracyComplaintRateExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.order_inaccuracy_complaint_rate import OrderInaccuracyComplaintRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_order_inaccuracy_complaint_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OrderInaccuracyComplaintRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"order_inaccuracy_complaint_rate_signal" in res.columns
    assert f"order_inaccuracy_complaint_rate_risk_score" in res.columns
    assert not res[f"order_inaccuracy_complaint_rate_signal"].isnull().any()

def test_order_inaccuracy_complaint_rate_empty_handling():
    extractor = OrderInaccuracyComplaintRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
