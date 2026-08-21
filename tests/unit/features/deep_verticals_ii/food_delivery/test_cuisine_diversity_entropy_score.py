# Comprehensive Unit Test for CuisineDiversityEntropyExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.cuisine_diversity_entropy_score import CuisineDiversityEntropyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cuisine_diversity_entropy_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CuisineDiversityEntropyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cuisine_diversity_entropy_score_signal" in res.columns
    assert f"cuisine_diversity_entropy_score_risk_score" in res.columns
    assert not res[f"cuisine_diversity_entropy_score_signal"].isnull().any()

def test_cuisine_diversity_entropy_score_empty_handling():
    extractor = CuisineDiversityEntropyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
