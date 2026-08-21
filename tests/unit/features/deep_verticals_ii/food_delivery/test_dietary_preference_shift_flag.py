# Comprehensive Unit Test for DietaryPreferenceShiftExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.dietary_preference_shift_flag import DietaryPreferenceShiftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_dietary_preference_shift_flag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DietaryPreferenceShiftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"dietary_preference_shift_flag_signal" in res.columns
    assert f"dietary_preference_shift_flag_risk_score" in res.columns
    assert not res[f"dietary_preference_shift_flag_signal"].isnull().any()

def test_dietary_preference_shift_flag_empty_handling():
    extractor = DietaryPreferenceShiftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
