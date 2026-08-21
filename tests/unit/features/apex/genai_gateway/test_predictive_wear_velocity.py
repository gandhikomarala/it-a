# Unit Test for PredictiveWearVelocityExtractor_Genaigateway (Generative AI API & Gateway Platform).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.genai_gateway.predictive_wear_velocity import PredictiveWearVelocityExtractor_Genaigateway
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_genai_gateway_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Genaigateway()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_genai_gateway_signal" in res.columns
    assert f"predictive_wear_velocity_genai_gateway_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_genai_gateway_signal"].isnull().any()

def test_predictive_wear_velocity_genai_gateway_empty():
    extractor = PredictiveWearVelocityExtractor_Genaigateway()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
