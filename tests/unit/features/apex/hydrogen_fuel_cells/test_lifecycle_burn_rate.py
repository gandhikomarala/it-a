# Unit Test for LifecycleBurnRateExtractor_Hydrogenfuelcells (Green Hydrogen Electrolyzer & Fuel Cells).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hydrogen_fuel_cells.lifecycle_burn_rate import LifecycleBurnRateExtractor_Hydrogenfuelcells
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_hydrogen_fuel_cells_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Hydrogenfuelcells()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_hydrogen_fuel_cells_signal" in res.columns
    assert f"lifecycle_burn_rate_hydrogen_fuel_cells_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_hydrogen_fuel_cells_signal"].isnull().any()

def test_lifecycle_burn_rate_hydrogen_fuel_cells_empty():
    extractor = LifecycleBurnRateExtractor_Hydrogenfuelcells()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
