# Unit Test for PowerUserConcentration (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.power_user_concentration import PowerUserConcentration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_power_user_concentration_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PowerUserConcentration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"power_user_concentration_signal" in res.columns
    assert f"power_user_concentration_risk_index" in res.columns
    assert not res[f"power_user_concentration_signal"].isnull().any()

def test_power_user_concentration_empty_handling():
    extractor = PowerUserConcentration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
