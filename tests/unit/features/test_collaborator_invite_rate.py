# Comprehensive Unit Test for CollaboratorInviteRateExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.collaborator_invite_rate import CollaboratorInviteRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_collaborator_invite_rate_instantiation():
    extractor = CollaboratorInviteRateExtractor()
    assert extractor.prefix == "collaborator_invite_rate"

def test_collaborator_invite_rate_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = CollaboratorInviteRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("collaborator_invite_rate_")]
    assert len(expected_cols) > 0

def test_collaborator_invite_rate_transform_empty():
    extractor = CollaboratorInviteRateExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
