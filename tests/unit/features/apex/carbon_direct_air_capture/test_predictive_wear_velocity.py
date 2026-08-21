# Unit Test for PredictiveWearVelocityExtractor_Carbondirectaircapture (Direct Air Carbon Capture & Sequestration).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_direct_air_capture.predictive_wear_velocity import PredictiveWearVelocityExtractor_Carbondirectaircapture
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_carbon_direct_air_capture_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Carbondirectaircapture()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_carbon_direct_air_capture_signal" in res.columns
    assert f"predictive_wear_velocity_carbon_direct_air_capture_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_carbon_direct_air_capture_signal"].isnull().any()

def test_predictive_wear_velocity_carbon_direct_air_capture_empty():
    extractor = PredictiveWearVelocityExtractor_Carbondirectaircapture()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
