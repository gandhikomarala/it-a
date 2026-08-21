# Unit Test for PredictiveWearVelocityExtractor_Digitalpathologyai (Whole Slide Imaging Digital Pathology AI).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.digital_pathology_ai.predictive_wear_velocity import PredictiveWearVelocityExtractor_Digitalpathologyai
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_digital_pathology_ai_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Digitalpathologyai()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_digital_pathology_ai_signal" in res.columns
    assert f"predictive_wear_velocity_digital_pathology_ai_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_digital_pathology_ai_signal"].isnull().any()

def test_predictive_wear_velocity_digital_pathology_ai_empty():
    extractor = PredictiveWearVelocityExtractor_Digitalpathologyai()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
