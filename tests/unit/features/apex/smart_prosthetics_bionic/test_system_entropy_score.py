# Unit Test for SystemEntropyScoreExtractor_Smartprostheticsbionic (Myoelectric Bionic Prosthetics Control).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_prosthetics_bionic.system_entropy_score import SystemEntropyScoreExtractor_Smartprostheticsbionic
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_smart_prosthetics_bionic_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Smartprostheticsbionic()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_smart_prosthetics_bionic_signal" in res.columns
    assert f"system_entropy_score_smart_prosthetics_bionic_risk_score" in res.columns
    assert not res[f"system_entropy_score_smart_prosthetics_bionic_signal"].isnull().any()

def test_system_entropy_score_smart_prosthetics_bionic_empty():
    extractor = SystemEntropyScoreExtractor_Smartprostheticsbionic()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
