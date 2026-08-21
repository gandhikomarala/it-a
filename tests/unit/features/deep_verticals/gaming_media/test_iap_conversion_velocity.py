# Comprehensive Unit Test for IAPConversionVelocityExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.iap_conversion_velocity import IAPConversionVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_iap_conversion_velocity_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = IAPConversionVelocityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"iap_conversion_velocity_signal" in res.columns
    assert f"iap_conversion_velocity_risk_score" in res.columns
    assert not res[f"iap_conversion_velocity_signal"].isnull().any()

def test_iap_conversion_velocity_empty():
    extractor = IAPConversionVelocityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
