# Comprehensive Unit Test for GuestReviewSentimentExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.guest_review_sentiment_score import GuestReviewSentimentExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_guest_review_sentiment_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = GuestReviewSentimentExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"guest_review_sentiment_score_signal" in res.columns
    assert f"guest_review_sentiment_score_risk_score" in res.columns
    assert not res[f"guest_review_sentiment_score_signal"].isnull().any()

def test_guest_review_sentiment_score_empty_handling():
    extractor = GuestReviewSentimentExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
