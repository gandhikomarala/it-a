# Comprehensive Unit Test for CourseCompletionVelocityExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.course_completion_velocity import CourseCompletionVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_course_completion_velocity_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CourseCompletionVelocityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"course_completion_velocity_signal" in res.columns
    assert f"course_completion_velocity_risk_score" in res.columns
    assert not res[f"course_completion_velocity_signal"].isnull().any()

def test_course_completion_velocity_empty():
    extractor = CourseCompletionVelocityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
