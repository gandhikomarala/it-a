# Unit Test for AccountResilienceFactorExtractor_Oilgaspipeline (Oil & Gas Pipeline Integrity).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.oil_gas_pipeline.account_resilience_factor import AccountResilienceFactorExtractor_Oilgaspipeline
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_oil_gas_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Oilgaspipeline()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_oil_gas_pipeline_signal" in res.columns
    assert f"account_resilience_factor_oil_gas_pipeline_risk_score" in res.columns
    assert not res[f"account_resilience_factor_oil_gas_pipeline_signal"].isnull().any()

def test_account_resilience_factor_oil_gas_pipeline_empty():
    extractor = AccountResilienceFactorExtractor_Oilgaspipeline()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
