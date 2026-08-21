# Unit Test for SystemEntropyScoreExtractor_Semiconductorphotolithography (EUV Semiconductor Photolithography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.semiconductor_photolithography.system_entropy_score import SystemEntropyScoreExtractor_Semiconductorphotolithography
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_semiconductor_photolithography_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Semiconductorphotolithography()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_semiconductor_photolithography_signal" in res.columns
    assert f"system_entropy_score_semiconductor_photolithography_risk_score" in res.columns
    assert not res[f"system_entropy_score_semiconductor_photolithography_signal"].isnull().any()

def test_system_entropy_score_semiconductor_photolithography_empty():
    extractor = SystemEntropyScoreExtractor_Semiconductorphotolithography()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
