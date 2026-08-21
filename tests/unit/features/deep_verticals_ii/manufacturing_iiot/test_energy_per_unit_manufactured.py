# Comprehensive Unit Test for EnergyPerUnitManufacturedExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.energy_per_unit_manufactured import EnergyPerUnitManufacturedExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_energy_per_unit_manufactured_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EnergyPerUnitManufacturedExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"energy_per_unit_manufactured_signal" in res.columns
    assert f"energy_per_unit_manufactured_risk_score" in res.columns
    assert not res[f"energy_per_unit_manufactured_signal"].isnull().any()

def test_energy_per_unit_manufactured_empty_handling():
    extractor = EnergyPerUnitManufacturedExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
