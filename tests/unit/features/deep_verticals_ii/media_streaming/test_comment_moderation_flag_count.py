# Comprehensive Unit Test for CommentModerationFlagCountExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.comment_moderation_flag_count import CommentModerationFlagCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_comment_moderation_flag_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CommentModerationFlagCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"comment_moderation_flag_count_signal" in res.columns
    assert f"comment_moderation_flag_count_risk_score" in res.columns
    assert not res[f"comment_moderation_flag_count_signal"].isnull().any()

def test_comment_moderation_flag_count_empty_handling():
    extractor = CommentModerationFlagCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
