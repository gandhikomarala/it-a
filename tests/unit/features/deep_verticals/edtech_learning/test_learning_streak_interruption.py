# Comprehensive Unit Test for LearningStreakInterruptionExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.learning_streak_interruption import LearningStreakInterruptionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_learning_streak_interruption_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LearningStreakInterruptionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"learning_streak_interruption_signal" in res.columns
    assert f"learning_streak_interruption_risk_score" in res.columns
    assert not res[f"learning_streak_interruption_signal"].isnull().any()

def test_learning_streak_interruption_empty():
    extractor = LearningStreakInterruptionExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
