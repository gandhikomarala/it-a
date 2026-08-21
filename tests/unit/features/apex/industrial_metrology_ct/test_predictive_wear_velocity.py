# Unit Test for PredictiveWearVelocityExtractor_Industrialmetrologyct (Industrial X-Ray Computed Tomography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.industrial_metrology_ct.predictive_wear_velocity import PredictiveWearVelocityExtractor_Industrialmetrologyct
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_industrial_metrology_ct_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Industrialmetrologyct()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_industrial_metrology_ct_signal" in res.columns
    assert f"predictive_wear_velocity_industrial_metrology_ct_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_industrial_metrology_ct_signal"].isnull().any()

def test_predictive_wear_velocity_industrial_metrology_ct_empty():
    extractor = PredictiveWearVelocityExtractor_Industrialmetrologyct()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
