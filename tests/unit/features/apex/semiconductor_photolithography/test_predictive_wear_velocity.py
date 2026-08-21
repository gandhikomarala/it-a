# Unit Test for PredictiveWearVelocityExtractor_Semiconductorphotolithography (EUV Semiconductor Photolithography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.semiconductor_photolithography.predictive_wear_velocity import PredictiveWearVelocityExtractor_Semiconductorphotolithography
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_semiconductor_photolithography_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Semiconductorphotolithography()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_semiconductor_photolithography_signal" in res.columns
    assert f"predictive_wear_velocity_semiconductor_photolithography_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_semiconductor_photolithography_signal"].isnull().any()

def test_predictive_wear_velocity_semiconductor_photolithography_empty():
    extractor = PredictiveWearVelocityExtractor_Semiconductorphotolithography()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
