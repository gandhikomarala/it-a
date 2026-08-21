# Comprehensive Unit Test for OutsideCounselBillingRateDriftExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.outside_counsel_billing_rate_drift import OutsideCounselBillingRateDriftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_outside_counsel_billing_rate_drift_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OutsideCounselBillingRateDriftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"outside_counsel_billing_rate_drift_signal" in res.columns
    assert f"outside_counsel_billing_rate_drift_risk_score" in res.columns
    assert not res[f"outside_counsel_billing_rate_drift_signal"].isnull().any()

def test_outside_counsel_billing_rate_drift_empty_handling():
    extractor = OutsideCounselBillingRateDriftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
