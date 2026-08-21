# Unit Test for VolatilityIndexScoreExtractor_Colocationdatacenter (Wholesale Colocation Datacenter Power).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.colocation_datacenter.volatility_index_score import VolatilityIndexScoreExtractor_Colocationdatacenter
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_colocation_datacenter_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Colocationdatacenter()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_colocation_datacenter_signal" in res.columns
    assert f"volatility_index_score_colocation_datacenter_risk_score" in res.columns
    assert not res[f"volatility_index_score_colocation_datacenter_signal"].isnull().any()

def test_volatility_index_score_colocation_datacenter_empty():
    extractor = VolatilityIndexScoreExtractor_Colocationdatacenter()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
