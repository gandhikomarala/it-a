# Unit Test for PredictiveWearVelocityExtractor_Timbersmartforestry (Autonomous Precision Forestry Harvesters).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.timber_smart_forestry.predictive_wear_velocity import PredictiveWearVelocityExtractor_Timbersmartforestry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_timber_smart_forestry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Timbersmartforestry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_timber_smart_forestry_signal" in res.columns
    assert f"predictive_wear_velocity_timber_smart_forestry_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_timber_smart_forestry_signal"].isnull().any()

def test_predictive_wear_velocity_timber_smart_forestry_empty():
    extractor = PredictiveWearVelocityExtractor_Timbersmartforestry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
