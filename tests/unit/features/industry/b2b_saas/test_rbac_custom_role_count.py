# Unit Test for RBACCustomRoleCount (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.rbac_custom_role_count import RBACCustomRoleCount
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_rbac_custom_role_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RBACCustomRoleCount()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"rbac_custom_role_count_signal" in res.columns
    assert f"rbac_custom_role_count_risk_index" in res.columns
    assert not res[f"rbac_custom_role_count_signal"].isnull().any()

def test_rbac_custom_role_count_empty_handling():
    extractor = RBACCustomRoleCount()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
