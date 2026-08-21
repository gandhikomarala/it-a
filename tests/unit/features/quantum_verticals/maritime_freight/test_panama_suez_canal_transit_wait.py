# Comprehensive Unit Test for CanalTransitWaitExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.panama_suez_canal_transit_wait import CanalTransitWaitExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_panama_suez_canal_transit_wait_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CanalTransitWaitExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"panama_suez_canal_transit_wait_signal" in res.columns
    assert f"panama_suez_canal_transit_wait_risk_score" in res.columns
    assert not res[f"panama_suez_canal_transit_wait_signal"].isnull().any()

def test_panama_suez_canal_transit_wait_empty_handling():
    extractor = CanalTransitWaitExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
