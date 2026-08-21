# Unit Test for NumberPortabilityInquiry (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.number_portability_inquiry import NumberPortabilityInquiry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_number_portability_inquiry_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NumberPortabilityInquiry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"number_portability_inquiry_signal" in res.columns
    assert f"number_portability_inquiry_risk_index" in res.columns
    assert not res[f"number_portability_inquiry_signal"].isnull().any()

def test_number_portability_inquiry_empty_handling():
    extractor = NumberPortabilityInquiry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
