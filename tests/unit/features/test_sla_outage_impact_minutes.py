# Comprehensive Unit Test for SLAOutageImpactMinutesExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.sla_outage_impact_minutes import SLAOutageImpactMinutesExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_sla_outage_impact_minutes_instantiation():
    extractor = SLAOutageImpactMinutesExtractor()
    assert extractor.prefix == "sla_outage_impact_minutes"

def test_sla_outage_impact_minutes_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = SLAOutageImpactMinutesExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("sla_outage_impact_minutes_")]
    assert len(expected_cols) > 0

def test_sla_outage_impact_minutes_transform_empty():
    extractor = SLAOutageImpactMinutesExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
