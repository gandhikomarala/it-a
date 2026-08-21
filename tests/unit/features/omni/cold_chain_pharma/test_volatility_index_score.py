# Unit Test for VolatilityIndexScoreExtractor_Coldchainpharma (Cold Chain Biopharma Logistics).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cold_chain_pharma.volatility_index_score import VolatilityIndexScoreExtractor_Coldchainpharma
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_cold_chain_pharma_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Coldchainpharma()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_cold_chain_pharma_signal" in res.columns
    assert f"volatility_index_score_cold_chain_pharma_risk_score" in res.columns
    assert not res[f"volatility_index_score_cold_chain_pharma_signal"].isnull().any()

def test_volatility_index_score_cold_chain_pharma_empty():
    extractor = VolatilityIndexScoreExtractor_Coldchainpharma()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
