# Unit Test for RetentionHealthIndexExtractor_Carbonaccountingesg (Enterprise Scope 1-2-3 Carbon Accounting).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.carbon_accounting_esg.retention_health_index import RetentionHealthIndexExtractor_Carbonaccountingesg
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_carbon_accounting_esg_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Carbonaccountingesg()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_carbon_accounting_esg_signal" in res.columns
    assert f"retention_health_index_carbon_accounting_esg_risk_score" in res.columns
    assert not res[f"retention_health_index_carbon_accounting_esg_signal"].isnull().any()

def test_retention_health_index_carbon_accounting_esg_empty():
    extractor = RetentionHealthIndexExtractor_Carbonaccountingesg()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
