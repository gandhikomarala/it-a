# Comprehensive Unit Test for EmailBounceRateExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.email_deliverability_bounce_rate import EmailBounceRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_email_deliverability_bounce_rate_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EmailBounceRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"email_deliverability_bounce_rate_signal" in res.columns
    assert f"email_deliverability_bounce_rate_risk_score" in res.columns
    assert not res[f"email_deliverability_bounce_rate_signal"].isnull().any()

def test_email_deliverability_bounce_rate_empty():
    extractor = EmailBounceRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
