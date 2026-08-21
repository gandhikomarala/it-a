# Unit Test for PredictiveWearVelocityExtractor_Spaceflightlaunch (Commercial Space Flight Launch Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.spaceflight_launch.predictive_wear_velocity import PredictiveWearVelocityExtractor_Spaceflightlaunch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_spaceflight_launch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Spaceflightlaunch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_spaceflight_launch_signal" in res.columns
    assert f"predictive_wear_velocity_spaceflight_launch_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_spaceflight_launch_signal"].isnull().any()

def test_predictive_wear_velocity_spaceflight_launch_empty():
    extractor = PredictiveWearVelocityExtractor_Spaceflightlaunch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
