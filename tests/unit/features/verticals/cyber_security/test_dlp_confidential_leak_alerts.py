# Unit Test for DLPLeakAlertsExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.dlp_confidential_leak_alerts import DLPLeakAlertsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_dlp_confidential_leak_alerts_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DLPLeakAlertsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"dlp_confidential_leak_alerts_signal" in res.columns
    assert f"dlp_confidential_leak_alerts_risk_score" in res.columns
    assert not res[f"dlp_confidential_leak_alerts_signal"].isnull().any()

def test_dlp_confidential_leak_alerts_empty_dataframe():
    extractor = DLPLeakAlertsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
