# Comprehensive Unit Test for CollaboratorRemovalRateExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.collaborator_removal_rate import CollaboratorRemovalRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_collaborator_removal_rate_instantiation():
    extractor = CollaboratorRemovalRateExtractor()
    assert extractor.prefix == "collaborator_removal_rate"

def test_collaborator_removal_rate_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = CollaboratorRemovalRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("collaborator_removal_rate_")]
    assert len(expected_cols) > 0

def test_collaborator_removal_rate_transform_empty():
    extractor = CollaboratorRemovalRateExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
