# Unit Test for VolatilityIndexScoreExtractor_Carbonaccountingesg (Enterprise Scope 1-2-3 Carbon Accounting).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.carbon_accounting_esg.volatility_index_score import VolatilityIndexScoreExtractor_Carbonaccountingesg
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_carbon_accounting_esg_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Carbonaccountingesg()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_carbon_accounting_esg_signal" in res.columns
    assert f"volatility_index_score_carbon_accounting_esg_risk_score" in res.columns
    assert not res[f"volatility_index_score_carbon_accounting_esg_signal"].isnull().any()

def test_volatility_index_score_carbon_accounting_esg_empty():
    extractor = VolatilityIndexScoreExtractor_Carbonaccountingesg()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
