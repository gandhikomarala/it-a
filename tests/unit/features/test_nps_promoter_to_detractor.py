# Comprehensive Unit Test for NPSPromoterToDetractorExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.nps_promoter_to_detractor import NPSPromoterToDetractorExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_nps_promoter_to_detractor_instantiation():
    extractor = NPSPromoterToDetractorExtractor()
    assert extractor.prefix == "nps_promoter_to_detractor"

def test_nps_promoter_to_detractor_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = NPSPromoterToDetractorExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("nps_promoter_to_detractor_")]
    assert len(expected_cols) > 0

def test_nps_promoter_to_detractor_transform_empty():
    extractor = NPSPromoterToDetractorExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
