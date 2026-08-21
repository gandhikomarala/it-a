# Unit Test for PredictiveWearVelocityExtractor_Batterygigafactoryquality (Lithium-Ion Battery Gigafactory Cell QC).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.battery_gigafactory_quality.predictive_wear_velocity import PredictiveWearVelocityExtractor_Batterygigafactoryquality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_battery_gigafactory_quality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Batterygigafactoryquality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_battery_gigafactory_quality_signal" in res.columns
    assert f"predictive_wear_velocity_battery_gigafactory_quality_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_battery_gigafactory_quality_signal"].isnull().any()

def test_predictive_wear_velocity_battery_gigafactory_quality_empty():
    extractor = PredictiveWearVelocityExtractor_Batterygigafactoryquality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
