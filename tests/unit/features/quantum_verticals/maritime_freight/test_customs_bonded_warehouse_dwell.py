# Comprehensive Unit Test for BondedWarehouseDwellExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.customs_bonded_warehouse_dwell import BondedWarehouseDwellExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_customs_bonded_warehouse_dwell_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BondedWarehouseDwellExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"customs_bonded_warehouse_dwell_signal" in res.columns
    assert f"customs_bonded_warehouse_dwell_risk_score" in res.columns
    assert not res[f"customs_bonded_warehouse_dwell_signal"].isnull().any()

def test_customs_bonded_warehouse_dwell_empty_handling():
    extractor = BondedWarehouseDwellExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
