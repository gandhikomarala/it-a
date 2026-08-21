# Comprehensive Unit Test for MunicipalCallVolumeSpikeExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.311_call_volume_spike_ratio import MunicipalCallVolumeSpikeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_311_call_volume_spike_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MunicipalCallVolumeSpikeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"311_call_volume_spike_ratio_signal" in res.columns
    assert f"311_call_volume_spike_ratio_risk_score" in res.columns
    assert not res[f"311_call_volume_spike_ratio_signal"].isnull().any()

def test_311_call_volume_spike_ratio_empty_handling():
    extractor = MunicipalCallVolumeSpikeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
