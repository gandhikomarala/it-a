# Comprehensive Unit Test for QuizRetryFrustrationExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.quiz_retry_frustration_index import QuizRetryFrustrationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quiz_retry_frustration_index_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = QuizRetryFrustrationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quiz_retry_frustration_index_signal" in res.columns
    assert f"quiz_retry_frustration_index_risk_score" in res.columns
    assert not res[f"quiz_retry_frustration_index_signal"].isnull().any()

def test_quiz_retry_frustration_index_empty():
    extractor = QuizRetryFrustrationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
