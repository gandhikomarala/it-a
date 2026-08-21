# Comprehensive Unit Test for SupportAgentTouchpointExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.support_agent_touchpoints import SupportAgentTouchpointExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_support_agent_touchpoints_instantiation():
    extractor = SupportAgentTouchpointExtractor()
    assert extractor.prefix == "support_agent_touchpoints"

def test_support_agent_touchpoints_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = SupportAgentTouchpointExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("support_agent_touchpoints_")]
    assert len(expected_cols) > 0

def test_support_agent_touchpoints_transform_empty():
    extractor = SupportAgentTouchpointExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
