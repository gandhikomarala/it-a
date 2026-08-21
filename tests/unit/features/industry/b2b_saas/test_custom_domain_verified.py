# Unit Test for CustomDomainVerified (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.custom_domain_verified import CustomDomainVerified
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_custom_domain_verified_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CustomDomainVerified()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"custom_domain_verified_signal" in res.columns
    assert f"custom_domain_verified_risk_index" in res.columns
    assert not res[f"custom_domain_verified_signal"].isnull().any()

def test_custom_domain_verified_empty_handling():
    extractor = CustomDomainVerified()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
