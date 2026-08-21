# Unit Test for SystemEntropyScoreExtractor_Autonomousdrones (Autonomous Commercial Drone Delivery).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.autonomous_drones.system_entropy_score import SystemEntropyScoreExtractor_Autonomousdrones
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_autonomous_drones_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Autonomousdrones()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_autonomous_drones_signal" in res.columns
    assert f"system_entropy_score_autonomous_drones_risk_score" in res.columns
    assert not res[f"system_entropy_score_autonomous_drones_signal"].isnull().any()

def test_system_entropy_score_autonomous_drones_empty():
    extractor = SystemEntropyScoreExtractor_Autonomousdrones()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
