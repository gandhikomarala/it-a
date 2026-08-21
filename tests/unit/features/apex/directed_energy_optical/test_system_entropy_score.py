# Unit Test for SystemEntropyScoreExtractor_Directedenergyoptical (High-Energy Laser Beam Control Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.directed_energy_optical.system_entropy_score import SystemEntropyScoreExtractor_Directedenergyoptical
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_directed_energy_optical_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Directedenergyoptical()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_directed_energy_optical_signal" in res.columns
    assert f"system_entropy_score_directed_energy_optical_risk_score" in res.columns
    assert not res[f"system_entropy_score_directed_energy_optical_signal"].isnull().any()

def test_system_entropy_score_directed_energy_optical_empty():
    extractor = SystemEntropyScoreExtractor_Directedenergyoptical()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
