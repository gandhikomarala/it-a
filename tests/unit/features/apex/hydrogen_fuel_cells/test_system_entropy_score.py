# Unit Test for SystemEntropyScoreExtractor_Hydrogenfuelcells (Green Hydrogen Electrolyzer & Fuel Cells).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hydrogen_fuel_cells.system_entropy_score import SystemEntropyScoreExtractor_Hydrogenfuelcells
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_hydrogen_fuel_cells_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Hydrogenfuelcells()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_hydrogen_fuel_cells_signal" in res.columns
    assert f"system_entropy_score_hydrogen_fuel_cells_risk_score" in res.columns
    assert not res[f"system_entropy_score_hydrogen_fuel_cells_signal"].isnull().any()

def test_system_entropy_score_hydrogen_fuel_cells_empty():
    extractor = SystemEntropyScoreExtractor_Hydrogenfuelcells()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
