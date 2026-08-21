# Unit Test for DNSTunnelingSuspicionScoreExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.dns_tunneling_suspicion_score import DNSTunnelingSuspicionScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_dns_tunneling_suspicion_score_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DNSTunnelingSuspicionScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"dns_tunneling_suspicion_score_signal" in res.columns
    assert f"dns_tunneling_suspicion_score_risk_score" in res.columns
    assert not res[f"dns_tunneling_suspicion_score_signal"].isnull().any()

def test_dns_tunneling_suspicion_score_empty_dataframe():
    extractor = DNSTunnelingSuspicionScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
