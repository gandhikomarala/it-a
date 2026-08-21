# Unit Test for QBRStatus (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.quarterly_business_review_status import QBRStatus
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quarterly_business_review_status_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = QBRStatus()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quarterly_business_review_status_signal" in res.columns
    assert f"quarterly_business_review_status_risk_index" in res.columns
    assert not res[f"quarterly_business_review_status_signal"].isnull().any()

def test_quarterly_business_review_status_empty_handling():
    extractor = QBRStatus()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
