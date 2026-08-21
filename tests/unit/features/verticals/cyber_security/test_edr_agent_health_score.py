# Unit Test for EDRAgentHealthScoreExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.edr_agent_health_score import EDRAgentHealthScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_edr_agent_health_score_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EDRAgentHealthScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"edr_agent_health_score_signal" in res.columns
    assert f"edr_agent_health_score_risk_score" in res.columns
    assert not res[f"edr_agent_health_score_signal"].isnull().any()

def test_edr_agent_health_score_empty_dataframe():
    extractor = EDRAgentHealthScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
