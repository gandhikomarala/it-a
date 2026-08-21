# Unit Test for PlasmaInstabilityGrowthExtractor_Solidoxidefuelcellstacks (High-Temperature SOFC Fuel Cell Stacks).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.solid_oxide_fuel_cell_stacks.plasma_instability_growth_rate import PlasmaInstabilityGrowthExtractor_Solidoxidefuelcellstacks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_plasma_instability_growth_rate_solid_oxide_fuel_cell_stacks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PlasmaInstabilityGrowthExtractor_Solidoxidefuelcellstacks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"plasma_instability_growth_rate_solid_oxide_fuel_cell_stacks_signal" in res.columns
    assert f"plasma_instability_growth_rate_solid_oxide_fuel_cell_stacks_risk_score" in res.columns
    assert not res[f"plasma_instability_growth_rate_solid_oxide_fuel_cell_stacks_signal"].isnull().any()

def test_plasma_instability_growth_rate_solid_oxide_fuel_cell_stacks_empty():
    extractor = PlasmaInstabilityGrowthExtractor_Solidoxidefuelcellstacks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
