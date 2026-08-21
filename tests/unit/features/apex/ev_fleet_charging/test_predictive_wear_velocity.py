# Unit Test for PredictiveWearVelocityExtractor_Evfleetcharging (Commercial Electric Vehicle Fleet Charging).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.ev_fleet_charging.predictive_wear_velocity import PredictiveWearVelocityExtractor_Evfleetcharging
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_ev_fleet_charging_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Evfleetcharging()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_ev_fleet_charging_signal" in res.columns
    assert f"predictive_wear_velocity_ev_fleet_charging_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_ev_fleet_charging_signal"].isnull().any()

def test_predictive_wear_velocity_ev_fleet_charging_empty():
    extractor = PredictiveWearVelocityExtractor_Evfleetcharging()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
