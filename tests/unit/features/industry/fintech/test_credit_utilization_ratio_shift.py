# Unit Test for CreditUtilizationShift (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.credit_utilization_ratio_shift import CreditUtilizationShift
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_credit_utilization_ratio_shift_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CreditUtilizationShift()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"credit_utilization_ratio_shift_signal" in res.columns
    assert f"credit_utilization_ratio_shift_risk_index" in res.columns
    assert not res[f"credit_utilization_ratio_shift_signal"].isnull().any()

def test_credit_utilization_ratio_shift_empty_handling():
    extractor = CreditUtilizationShift()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
