# Comprehensive Unit Test for BrokerCommissionChurnFlagExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.broker_commission_churn_flag import BrokerCommissionChurnFlagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_broker_commission_churn_flag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BrokerCommissionChurnFlagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"broker_commission_churn_flag_signal" in res.columns
    assert f"broker_commission_churn_flag_risk_score" in res.columns
    assert not res[f"broker_commission_churn_flag_signal"].isnull().any()

def test_broker_commission_churn_flag_empty_handling():
    extractor = BrokerCommissionChurnFlagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
