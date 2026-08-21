# Unit Test for SystemEntropyScoreExtractor_Modularnuclearsmr (Small Modular Nuclear Reactor (SMR) Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.modular_nuclear_smr.system_entropy_score import SystemEntropyScoreExtractor_Modularnuclearsmr
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_modular_nuclear_smr_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Modularnuclearsmr()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_modular_nuclear_smr_signal" in res.columns
    assert f"system_entropy_score_modular_nuclear_smr_risk_score" in res.columns
    assert not res[f"system_entropy_score_modular_nuclear_smr_signal"].isnull().any()

def test_system_entropy_score_modular_nuclear_smr_empty():
    extractor = SystemEntropyScoreExtractor_Modularnuclearsmr()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
