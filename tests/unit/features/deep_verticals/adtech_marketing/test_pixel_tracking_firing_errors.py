# Comprehensive Unit Test for PixelFiringErrorExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.pixel_tracking_firing_errors import PixelFiringErrorExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_pixel_tracking_firing_errors_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PixelFiringErrorExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"pixel_tracking_firing_errors_signal" in res.columns
    assert f"pixel_tracking_firing_errors_risk_score" in res.columns
    assert not res[f"pixel_tracking_firing_errors_signal"].isnull().any()

def test_pixel_tracking_firing_errors_empty():
    extractor = PixelFiringErrorExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
