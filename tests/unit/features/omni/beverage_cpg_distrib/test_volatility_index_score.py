# Unit Test for VolatilityIndexScoreExtractor_Beveragecpgdistrib (Beverage CPG Direct Store Delivery).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.beverage_cpg_distrib.volatility_index_score import VolatilityIndexScoreExtractor_Beveragecpgdistrib
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_beverage_cpg_distrib_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Beveragecpgdistrib()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_beverage_cpg_distrib_signal" in res.columns
    assert f"volatility_index_score_beverage_cpg_distrib_risk_score" in res.columns
    assert not res[f"volatility_index_score_beverage_cpg_distrib_signal"].isnull().any()

def test_volatility_index_score_beverage_cpg_distrib_empty():
    extractor = VolatilityIndexScoreExtractor_Beveragecpgdistrib()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
