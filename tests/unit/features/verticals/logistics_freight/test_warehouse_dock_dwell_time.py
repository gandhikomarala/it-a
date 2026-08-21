# Unit Test for WarehouseDockDwellTimeExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.warehouse_dock_dwell_time import WarehouseDockDwellTimeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_warehouse_dock_dwell_time_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = WarehouseDockDwellTimeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"warehouse_dock_dwell_time_signal" in res.columns
    assert f"warehouse_dock_dwell_time_risk_score" in res.columns
    assert not res[f"warehouse_dock_dwell_time_signal"].isnull().any()

def test_warehouse_dock_dwell_time_empty_dataframe():
    extractor = WarehouseDockDwellTimeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
