# Comprehensive Unit Test for VirtualTourDwellExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.virtual_3d_tour_dwell_seconds import VirtualTourDwellExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_virtual_3d_tour_dwell_seconds_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VirtualTourDwellExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"virtual_3d_tour_dwell_seconds_signal" in res.columns
    assert f"virtual_3d_tour_dwell_seconds_risk_score" in res.columns
    assert not res[f"virtual_3d_tour_dwell_seconds_signal"].isnull().any()

def test_virtual_3d_tour_dwell_seconds_empty():
    extractor = VirtualTourDwellExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
