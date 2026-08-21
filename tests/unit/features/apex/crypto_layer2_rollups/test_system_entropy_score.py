# Unit Test for SystemEntropyScoreExtractor_Cryptolayer2Rollups (Zero-Knowledge Ethereum Layer-2 Rollups).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.crypto_layer2_rollups.system_entropy_score import SystemEntropyScoreExtractor_Cryptolayer2Rollups
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_crypto_layer2_rollups_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Cryptolayer2Rollups()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_crypto_layer2_rollups_signal" in res.columns
    assert f"system_entropy_score_crypto_layer2_rollups_risk_score" in res.columns
    assert not res[f"system_entropy_score_crypto_layer2_rollups_signal"].isnull().any()

def test_system_entropy_score_crypto_layer2_rollups_empty():
    extractor = SystemEntropyScoreExtractor_Cryptolayer2Rollups()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
