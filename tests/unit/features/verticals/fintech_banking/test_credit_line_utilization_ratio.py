# Unit Test for CreditLineUtilizationRatioExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.credit_line_utilization_ratio import CreditLineUtilizationRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_credit_line_utilization_ratio_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CreditLineUtilizationRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"credit_line_utilization_ratio_signal" in res.columns
    assert f"credit_line_utilization_ratio_risk_score" in res.columns
    assert not res[f"credit_line_utilization_ratio_signal"].isnull().any()

def test_credit_line_utilization_ratio_empty_dataframe():
    extractor = CreditLineUtilizationRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
