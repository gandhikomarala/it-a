# Unit Test for PredictiveWearVelocityExtractor_Hydrogenfuelcells (Green Hydrogen Electrolyzer & Fuel Cells).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hydrogen_fuel_cells.predictive_wear_velocity import PredictiveWearVelocityExtractor_Hydrogenfuelcells
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_hydrogen_fuel_cells_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Hydrogenfuelcells()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_hydrogen_fuel_cells_signal" in res.columns
    assert f"predictive_wear_velocity_hydrogen_fuel_cells_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_hydrogen_fuel_cells_signal"].isnull().any()

def test_predictive_wear_velocity_hydrogen_fuel_cells_empty():
    extractor = PredictiveWearVelocityExtractor_Hydrogenfuelcells()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
