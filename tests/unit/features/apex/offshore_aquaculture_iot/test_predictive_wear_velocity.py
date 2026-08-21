# Unit Test for PredictiveWearVelocityExtractor_Offshoreaquacultureiot (Open-Ocean Smart Aquaculture Cages).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.offshore_aquaculture_iot.predictive_wear_velocity import PredictiveWearVelocityExtractor_Offshoreaquacultureiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_offshore_aquaculture_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Offshoreaquacultureiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_offshore_aquaculture_iot_signal" in res.columns
    assert f"predictive_wear_velocity_offshore_aquaculture_iot_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_offshore_aquaculture_iot_signal"].isnull().any()

def test_predictive_wear_velocity_offshore_aquaculture_iot_empty():
    extractor = PredictiveWearVelocityExtractor_Offshoreaquacultureiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
