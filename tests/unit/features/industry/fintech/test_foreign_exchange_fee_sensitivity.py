# Unit Test for FXFeeSensitivity (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.foreign_exchange_fee_sensitivity import FXFeeSensitivity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_foreign_exchange_fee_sensitivity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FXFeeSensitivity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"foreign_exchange_fee_sensitivity_signal" in res.columns
    assert f"foreign_exchange_fee_sensitivity_risk_index" in res.columns
    assert not res[f"foreign_exchange_fee_sensitivity_signal"].isnull().any()

def test_foreign_exchange_fee_sensitivity_empty_handling():
    extractor = FXFeeSensitivity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
