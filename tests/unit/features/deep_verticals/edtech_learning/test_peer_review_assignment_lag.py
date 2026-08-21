# Comprehensive Unit Test for PeerReviewAssignmentLagExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.peer_review_assignment_lag import PeerReviewAssignmentLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_peer_review_assignment_lag_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PeerReviewAssignmentLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"peer_review_assignment_lag_signal" in res.columns
    assert f"peer_review_assignment_lag_risk_score" in res.columns
    assert not res[f"peer_review_assignment_lag_signal"].isnull().any()

def test_peer_review_assignment_lag_empty():
    extractor = PeerReviewAssignmentLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
