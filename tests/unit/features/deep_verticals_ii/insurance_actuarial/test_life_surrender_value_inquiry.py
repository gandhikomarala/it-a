# Comprehensive Unit Test for LifeSurrenderValueInquiryExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.life_surrender_value_inquiry import LifeSurrenderValueInquiryExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_life_surrender_value_inquiry_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifeSurrenderValueInquiryExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"life_surrender_value_inquiry_signal" in res.columns
    assert f"life_surrender_value_inquiry_risk_score" in res.columns
    assert not res[f"life_surrender_value_inquiry_signal"].isnull().any()

def test_life_surrender_value_inquiry_empty_handling():
    extractor = LifeSurrenderValueInquiryExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
