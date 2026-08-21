# Comprehensive Unit Test for PaywallMeterExhaustionRateExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.paywall_meter_exhaustion_rate import PaywallMeterExhaustionRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_paywall_meter_exhaustion_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PaywallMeterExhaustionRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"paywall_meter_exhaustion_rate_signal" in res.columns
    assert f"paywall_meter_exhaustion_rate_risk_score" in res.columns
    assert not res[f"paywall_meter_exhaustion_rate_signal"].isnull().any()

def test_paywall_meter_exhaustion_rate_empty_handling():
    extractor = PaywallMeterExhaustionRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
