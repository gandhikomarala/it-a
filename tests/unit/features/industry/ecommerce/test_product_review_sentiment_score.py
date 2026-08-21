# Unit Test for ProductReviewSentimentScore (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.product_review_sentiment_score import ProductReviewSentimentScore
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_product_review_sentiment_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ProductReviewSentimentScore()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"product_review_sentiment_score_signal" in res.columns
    assert f"product_review_sentiment_score_risk_index" in res.columns
    assert not res[f"product_review_sentiment_score_signal"].isnull().any()

def test_product_review_sentiment_score_empty_handling():
    extractor = ProductReviewSentimentScore()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
