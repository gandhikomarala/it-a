# Comprehensive Unit Test for TrademarkRenewalCountdownDaysExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.trademark_renewal_countdown_days import TrademarkRenewalCountdownDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_trademark_renewal_countdown_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TrademarkRenewalCountdownDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"trademark_renewal_countdown_days_signal" in res.columns
    assert f"trademark_renewal_countdown_days_risk_score" in res.columns
    assert not res[f"trademark_renewal_countdown_days_signal"].isnull().any()

def test_trademark_renewal_countdown_days_empty_handling():
    extractor = TrademarkRenewalCountdownDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
