# Comprehensive Unit Test for ReviewSubmissionDelayExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.performance_review_submission_delay import ReviewSubmissionDelayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_performance_review_submission_delay_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ReviewSubmissionDelayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"performance_review_submission_delay_signal" in res.columns
    assert f"performance_review_submission_delay_risk_score" in res.columns
    assert not res[f"performance_review_submission_delay_signal"].isnull().any()

def test_performance_review_submission_delay_empty():
    extractor = ReviewSubmissionDelayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
