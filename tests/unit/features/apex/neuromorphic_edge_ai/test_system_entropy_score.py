# Unit Test for SystemEntropyScoreExtractor_Neuromorphicedgeai (Neuromorphic Spiking Neural Network Edge AI).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.neuromorphic_edge_ai.system_entropy_score import SystemEntropyScoreExtractor_Neuromorphicedgeai
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_neuromorphic_edge_ai_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Neuromorphicedgeai()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_neuromorphic_edge_ai_signal" in res.columns
    assert f"system_entropy_score_neuromorphic_edge_ai_risk_score" in res.columns
    assert not res[f"system_entropy_score_neuromorphic_edge_ai_signal"].isnull().any()

def test_system_entropy_score_neuromorphic_edge_ai_empty():
    extractor = SystemEntropyScoreExtractor_Neuromorphicedgeai()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
