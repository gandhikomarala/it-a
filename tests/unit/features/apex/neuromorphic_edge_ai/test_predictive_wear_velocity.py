# Unit Test for PredictiveWearVelocityExtractor_Neuromorphicedgeai (Neuromorphic Spiking Neural Network Edge AI).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.neuromorphic_edge_ai.predictive_wear_velocity import PredictiveWearVelocityExtractor_Neuromorphicedgeai
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_neuromorphic_edge_ai_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Neuromorphicedgeai()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_neuromorphic_edge_ai_signal" in res.columns
    assert f"predictive_wear_velocity_neuromorphic_edge_ai_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_neuromorphic_edge_ai_signal"].isnull().any()

def test_predictive_wear_velocity_neuromorphic_edge_ai_empty():
    extractor = PredictiveWearVelocityExtractor_Neuromorphicedgeai()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
