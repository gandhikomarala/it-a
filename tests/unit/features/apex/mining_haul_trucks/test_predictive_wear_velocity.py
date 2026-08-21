# Unit Test for PredictiveWearVelocityExtractor_Mininghaultrucks (Autonomous Mining Haul Truck Fleets).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.mining_haul_trucks.predictive_wear_velocity import PredictiveWearVelocityExtractor_Mininghaultrucks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_mining_haul_trucks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Mininghaultrucks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_mining_haul_trucks_signal" in res.columns
    assert f"predictive_wear_velocity_mining_haul_trucks_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_mining_haul_trucks_signal"].isnull().any()

def test_predictive_wear_velocity_mining_haul_trucks_empty():
    extractor = PredictiveWearVelocityExtractor_Mininghaultrucks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
