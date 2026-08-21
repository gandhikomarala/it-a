# Comprehensive Unit Test for CIIRatingExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.maritime_carbon_intensity_cii_rating import CIIRatingExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_maritime_carbon_intensity_cii_rating_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CIIRatingExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"maritime_carbon_intensity_cii_rating_signal" in res.columns
    assert f"maritime_carbon_intensity_cii_rating_risk_score" in res.columns
    assert not res[f"maritime_carbon_intensity_cii_rating_signal"].isnull().any()

def test_maritime_carbon_intensity_cii_rating_empty_handling():
    extractor = CIIRatingExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
