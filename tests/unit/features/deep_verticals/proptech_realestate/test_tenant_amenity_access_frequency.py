# Comprehensive Unit Test for AmenityAccessFrequencyExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.tenant_amenity_access_frequency import AmenityAccessFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_tenant_amenity_access_frequency_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AmenityAccessFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"tenant_amenity_access_frequency_signal" in res.columns
    assert f"tenant_amenity_access_frequency_risk_score" in res.columns
    assert not res[f"tenant_amenity_access_frequency_signal"].isnull().any()

def test_tenant_amenity_access_frequency_empty():
    extractor = AmenityAccessFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
