# Unit Test for PredictiveWearVelocityExtractor_Airtrafficadsb (NextGen Air Traffic Control ADS-B Radar).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.air_traffic_ads_b.predictive_wear_velocity import PredictiveWearVelocityExtractor_Airtrafficadsb
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_air_traffic_ads_b_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Airtrafficadsb()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_air_traffic_ads_b_signal" in res.columns
    assert f"predictive_wear_velocity_air_traffic_ads_b_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_air_traffic_ads_b_signal"].isnull().any()

def test_predictive_wear_velocity_air_traffic_ads_b_empty():
    extractor = PredictiveWearVelocityExtractor_Airtrafficadsb()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
