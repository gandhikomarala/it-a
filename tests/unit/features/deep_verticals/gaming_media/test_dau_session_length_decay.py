# Comprehensive Unit Test for DAUSessionLengthDecayExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.dau_session_length_decay import DAUSessionLengthDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_dau_session_length_decay_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DAUSessionLengthDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"dau_session_length_decay_signal" in res.columns
    assert f"dau_session_length_decay_risk_score" in res.columns
    assert not res[f"dau_session_length_decay_signal"].isnull().any()

def test_dau_session_length_decay_empty():
    extractor = DAUSessionLengthDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
