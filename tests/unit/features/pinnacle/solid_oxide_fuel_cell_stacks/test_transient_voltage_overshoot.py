# Unit Test for TransientVoltageOvershootExtractor_Solidoxidefuelcellstacks (High-Temperature SOFC Fuel Cell Stacks).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.solid_oxide_fuel_cell_stacks.transient_voltage_overshoot import TransientVoltageOvershootExtractor_Solidoxidefuelcellstacks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_transient_voltage_overshoot_solid_oxide_fuel_cell_stacks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TransientVoltageOvershootExtractor_Solidoxidefuelcellstacks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"transient_voltage_overshoot_solid_oxide_fuel_cell_stacks_signal" in res.columns
    assert f"transient_voltage_overshoot_solid_oxide_fuel_cell_stacks_risk_score" in res.columns
    assert not res[f"transient_voltage_overshoot_solid_oxide_fuel_cell_stacks_signal"].isnull().any()

def test_transient_voltage_overshoot_solid_oxide_fuel_cell_stacks_empty():
    extractor = TransientVoltageOvershootExtractor_Solidoxidefuelcellstacks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
