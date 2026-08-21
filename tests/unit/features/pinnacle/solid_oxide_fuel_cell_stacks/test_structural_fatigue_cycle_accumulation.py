# Unit Test for FatigueCycleAccumulationExtractor_Solidoxidefuelcellstacks (High-Temperature SOFC Fuel Cell Stacks).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.solid_oxide_fuel_cell_stacks.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Solidoxidefuelcellstacks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_solid_oxide_fuel_cell_stacks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Solidoxidefuelcellstacks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_solid_oxide_fuel_cell_stacks_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_solid_oxide_fuel_cell_stacks_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_solid_oxide_fuel_cell_stacks_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_solid_oxide_fuel_cell_stacks_empty():
    extractor = FatigueCycleAccumulationExtractor_Solidoxidefuelcellstacks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
