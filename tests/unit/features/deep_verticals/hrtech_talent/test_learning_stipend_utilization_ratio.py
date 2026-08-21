# Comprehensive Unit Test for LearningStipendUtilizationExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.learning_stipend_utilization_ratio import LearningStipendUtilizationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_learning_stipend_utilization_ratio_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LearningStipendUtilizationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"learning_stipend_utilization_ratio_signal" in res.columns
    assert f"learning_stipend_utilization_ratio_risk_score" in res.columns
    assert not res[f"learning_stipend_utilization_ratio_signal"].isnull().any()

def test_learning_stipend_utilization_ratio_empty():
    extractor = LearningStipendUtilizationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
