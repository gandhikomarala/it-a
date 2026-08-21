# Unit Test for SystemEntropyScoreExtractor_Superconductingmagnetsmri (Ultra-High Field 7T MRI Superconducting Magnets).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.superconducting_magnets_mri.system_entropy_score import SystemEntropyScoreExtractor_Superconductingmagnetsmri
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_superconducting_magnets_mri_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Superconductingmagnetsmri()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_superconducting_magnets_mri_signal" in res.columns
    assert f"system_entropy_score_superconducting_magnets_mri_risk_score" in res.columns
    assert not res[f"system_entropy_score_superconducting_magnets_mri_signal"].isnull().any()

def test_system_entropy_score_superconducting_magnets_mri_empty():
    extractor = SystemEntropyScoreExtractor_Superconductingmagnetsmri()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
