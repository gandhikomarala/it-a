# Comprehensive Unit Test for PublicMeetingSentimentExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.public_meeting_comment_sentiment import PublicMeetingSentimentExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_public_meeting_comment_sentiment_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PublicMeetingSentimentExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"public_meeting_comment_sentiment_signal" in res.columns
    assert f"public_meeting_comment_sentiment_risk_score" in res.columns
    assert not res[f"public_meeting_comment_sentiment_signal"].isnull().any()

def test_public_meeting_comment_sentiment_empty_handling():
    extractor = PublicMeetingSentimentExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
