# Comprehensive Unit Test for OpenDataAPIQueryVolumeExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.open_data_api_query_volume import OpenDataAPIQueryVolumeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_open_data_api_query_volume_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OpenDataAPIQueryVolumeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"open_data_api_query_volume_signal" in res.columns
    assert f"open_data_api_query_volume_risk_score" in res.columns
    assert not res[f"open_data_api_query_volume_signal"].isnull().any()

def test_open_data_api_query_volume_empty_handling():
    extractor = OpenDataAPIQueryVolumeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
