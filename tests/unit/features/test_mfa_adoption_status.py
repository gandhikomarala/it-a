# Comprehensive Unit Test for MFAAdoptionStatusExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.mfa_adoption_status import MFAAdoptionStatusExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_mfa_adoption_status_instantiation():
    extractor = MFAAdoptionStatusExtractor()
    assert extractor.prefix == "mfa_adoption_status"

def test_mfa_adoption_status_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = MFAAdoptionStatusExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("mfa_adoption_status_")]
    assert len(expected_cols) > 0

def test_mfa_adoption_status_transform_empty():
    extractor = MFAAdoptionStatusExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
