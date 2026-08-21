# Unit Test for VolatilityIndexScoreExtractor_Corporatemandaduediligence (Corporate M&A Virtual Data Room).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.corporate_m_and_a_due_diligence.volatility_index_score import VolatilityIndexScoreExtractor_Corporatemandaduediligence
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_corporate_m_and_a_due_diligence_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Corporatemandaduediligence()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_corporate_m_and_a_due_diligence_signal" in res.columns
    assert f"volatility_index_score_corporate_m_and_a_due_diligence_risk_score" in res.columns
    assert not res[f"volatility_index_score_corporate_m_and_a_due_diligence_signal"].isnull().any()

def test_volatility_index_score_corporate_m_and_a_due_diligence_empty():
    extractor = VolatilityIndexScoreExtractor_Corporatemandaduediligence()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
