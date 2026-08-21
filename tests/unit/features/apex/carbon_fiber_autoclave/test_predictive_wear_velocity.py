# Unit Test for PredictiveWearVelocityExtractor_Carbonfiberautoclave (Aerospace Carbon Fiber Composite Autoclaves).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_fiber_autoclave.predictive_wear_velocity import PredictiveWearVelocityExtractor_Carbonfiberautoclave
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_carbon_fiber_autoclave_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Carbonfiberautoclave()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_carbon_fiber_autoclave_signal" in res.columns
    assert f"predictive_wear_velocity_carbon_fiber_autoclave_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_carbon_fiber_autoclave_signal"].isnull().any()

def test_predictive_wear_velocity_carbon_fiber_autoclave_empty():
    extractor = PredictiveWearVelocityExtractor_Carbonfiberautoclave()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
