# Comprehensive Unit Test for LegalHoldAcknowledgmentLagExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.legal_hold_acknowledgment_lag import LegalHoldAcknowledgmentLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_legal_hold_acknowledgment_lag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LegalHoldAcknowledgmentLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"legal_hold_acknowledgment_lag_signal" in res.columns
    assert f"legal_hold_acknowledgment_lag_risk_score" in res.columns
    assert not res[f"legal_hold_acknowledgment_lag_signal"].isnull().any()

def test_legal_hold_acknowledgment_lag_empty_handling():
    extractor = LegalHoldAcknowledgmentLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
