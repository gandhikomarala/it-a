# Comprehensive Unit Test for SessionDurationDecayExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.session_duration_decay import SessionDurationDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_session_duration_decay_instantiation():
    extractor = SessionDurationDecayExtractor()
    assert extractor.prefix == "session_duration_decay"

def test_session_duration_decay_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = SessionDurationDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("session_duration_decay_")]
    assert len(expected_cols) > 0

def test_session_duration_decay_transform_empty():
    extractor = SessionDurationDecayExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
