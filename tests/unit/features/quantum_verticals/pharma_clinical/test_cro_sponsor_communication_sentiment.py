# Comprehensive Unit Test for CROSponsorSentimentExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.cro_sponsor_communication_sentiment import CROSponsorSentimentExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cro_sponsor_communication_sentiment_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CROSponsorSentimentExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cro_sponsor_communication_sentiment_signal" in res.columns
    assert f"cro_sponsor_communication_sentiment_risk_score" in res.columns
    assert not res[f"cro_sponsor_communication_sentiment_signal"].isnull().any()

def test_cro_sponsor_communication_sentiment_empty_handling():
    extractor = CROSponsorSentimentExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
