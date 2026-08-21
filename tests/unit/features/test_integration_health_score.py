# Comprehensive Unit Test for IntegrationHealthScoreExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.integration_health_score import IntegrationHealthScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_integration_health_score_instantiation():
    extractor = IntegrationHealthScoreExtractor()
    assert extractor.prefix == "integration_health_score"

def test_integration_health_score_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = IntegrationHealthScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("integration_health_score_")]
    assert len(expected_cols) > 0

def test_integration_health_score_transform_empty():
    extractor = IntegrationHealthScoreExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
