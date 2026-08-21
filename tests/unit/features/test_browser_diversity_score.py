# Comprehensive Unit Test for BrowserDiversityScoreExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.browser_diversity_score import BrowserDiversityScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_browser_diversity_score_instantiation():
    extractor = BrowserDiversityScoreExtractor()
    assert extractor.prefix == "browser_diversity_score"

def test_browser_diversity_score_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = BrowserDiversityScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("browser_diversity_score_")]
    assert len(expected_cols) > 0

def test_browser_diversity_score_transform_empty():
    extractor = BrowserDiversityScoreExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
