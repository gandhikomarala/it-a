# Comprehensive Unit Test for ForumDiscussionRepliesExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.forum_discussion_replies_count import ForumDiscussionRepliesExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_forum_discussion_replies_count_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ForumDiscussionRepliesExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"forum_discussion_replies_count_signal" in res.columns
    assert f"forum_discussion_replies_count_risk_score" in res.columns
    assert not res[f"forum_discussion_replies_count_signal"].isnull().any()

def test_forum_discussion_replies_count_empty():
    extractor = ForumDiscussionRepliesExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
