# Unit Test for VolatilityIndexScoreExtractor_Legalippatents (Patent Prosecution & IP Portfolio).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.legal_ip_patents.volatility_index_score import VolatilityIndexScoreExtractor_Legalippatents
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_legal_ip_patents_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Legalippatents()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_legal_ip_patents_signal" in res.columns
    assert f"volatility_index_score_legal_ip_patents_risk_score" in res.columns
    assert not res[f"volatility_index_score_legal_ip_patents_signal"].isnull().any()

def test_volatility_index_score_legal_ip_patents_empty():
    extractor = VolatilityIndexScoreExtractor_Legalippatents()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
