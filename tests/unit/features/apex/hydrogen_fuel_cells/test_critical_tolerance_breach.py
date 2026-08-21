# Unit Test for CriticalToleranceBreachExtractor_Hydrogenfuelcells (Green Hydrogen Electrolyzer & Fuel Cells).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hydrogen_fuel_cells.critical_tolerance_breach import CriticalToleranceBreachExtractor_Hydrogenfuelcells
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_hydrogen_fuel_cells_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Hydrogenfuelcells()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_hydrogen_fuel_cells_signal" in res.columns
    assert f"critical_tolerance_breach_hydrogen_fuel_cells_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_hydrogen_fuel_cells_signal"].isnull().any()

def test_critical_tolerance_breach_hydrogen_fuel_cells_empty():
    extractor = CriticalToleranceBreachExtractor_Hydrogenfuelcells()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
