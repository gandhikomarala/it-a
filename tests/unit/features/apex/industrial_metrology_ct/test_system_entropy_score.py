# Unit Test for SystemEntropyScoreExtractor_Industrialmetrologyct (Industrial X-Ray Computed Tomography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.industrial_metrology_ct.system_entropy_score import SystemEntropyScoreExtractor_Industrialmetrologyct
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_industrial_metrology_ct_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Industrialmetrologyct()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_industrial_metrology_ct_signal" in res.columns
    assert f"system_entropy_score_industrial_metrology_ct_risk_score" in res.columns
    assert not res[f"system_entropy_score_industrial_metrology_ct_signal"].isnull().any()

def test_system_entropy_score_industrial_metrology_ct_empty():
    extractor = SystemEntropyScoreExtractor_Industrialmetrologyct()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
