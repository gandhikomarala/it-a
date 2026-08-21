# Unit Test for EfficiencyDegradationPaceExtractor_Hydrogenfuelcells (Green Hydrogen Electrolyzer & Fuel Cells).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hydrogen_fuel_cells.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Hydrogenfuelcells
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_hydrogen_fuel_cells_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Hydrogenfuelcells()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_hydrogen_fuel_cells_signal" in res.columns
    assert f"efficiency_degradation_pace_hydrogen_fuel_cells_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_hydrogen_fuel_cells_signal"].isnull().any()

def test_efficiency_degradation_pace_hydrogen_fuel_cells_empty():
    extractor = EfficiencyDegradationPaceExtractor_Hydrogenfuelcells()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
