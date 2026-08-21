# Comprehensive Unit Test for ColdFoodRefundCountExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.cold_food_refund_request_count import ColdFoodRefundCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cold_food_refund_request_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ColdFoodRefundCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cold_food_refund_request_count_signal" in res.columns
    assert f"cold_food_refund_request_count_risk_score" in res.columns
    assert not res[f"cold_food_refund_request_count_signal"].isnull().any()

def test_cold_food_refund_request_count_empty_handling():
    extractor = ColdFoodRefundCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
