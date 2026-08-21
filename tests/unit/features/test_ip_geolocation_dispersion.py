# Comprehensive Unit Test for IPGeolocationDispersionExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.ip_geolocation_dispersion import IPGeolocationDispersionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ip_geolocation_dispersion_instantiation():
    extractor = IPGeolocationDispersionExtractor()
    assert extractor.prefix == "ip_geolocation_dispersion"

def test_ip_geolocation_dispersion_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = IPGeolocationDispersionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("ip_geolocation_dispersion_")]
    assert len(expected_cols) > 0

def test_ip_geolocation_dispersion_transform_empty():
    extractor = IPGeolocationDispersionExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
