# Comprehensive Unit Test for CourierTipPercentageTrendExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.courier_tip_percentage_trend import CourierTipPercentageTrendExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_courier_tip_percentage_trend_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CourierTipPercentageTrendExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"courier_tip_percentage_trend_signal" in res.columns
    assert f"courier_tip_percentage_trend_risk_score" in res.columns
    assert not res[f"courier_tip_percentage_trend_signal"].isnull().any()

def test_courier_tip_percentage_trend_empty_handling():
    extractor = CourierTipPercentageTrendExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
