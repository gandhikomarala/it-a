# Comprehensive Unit Test for TenantSatisfactionNPSDeltaExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.tenant_satisfaction_nps_delta import TenantSatisfactionNPSDeltaExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_tenant_satisfaction_nps_delta_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TenantSatisfactionNPSDeltaExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"tenant_satisfaction_nps_delta_signal" in res.columns
    assert f"tenant_satisfaction_nps_delta_risk_score" in res.columns
    assert not res[f"tenant_satisfaction_nps_delta_signal"].isnull().any()

def test_tenant_satisfaction_nps_delta_empty():
    extractor = TenantSatisfactionNPSDeltaExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
