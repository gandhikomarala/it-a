# Comprehensive Unit Test for LeadScoreDriftExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.lead_scoring_model_drift_index import LeadScoreDriftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lead_scoring_model_drift_index_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LeadScoreDriftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lead_scoring_model_drift_index_signal" in res.columns
    assert f"lead_scoring_model_drift_index_risk_score" in res.columns
    assert not res[f"lead_scoring_model_drift_index_signal"].isnull().any()

def test_lead_scoring_model_drift_index_empty():
    extractor = LeadScoreDriftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
