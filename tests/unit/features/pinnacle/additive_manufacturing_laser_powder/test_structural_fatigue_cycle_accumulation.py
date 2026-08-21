# Unit Test for FatigueCycleAccumulationExtractor_Additivemanufacturinglaserpowder (LPBF Metal Additive Manufacturing).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.additive_manufacturing_laser_powder.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Additivemanufacturinglaserpowder
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_additive_manufacturing_laser_powder_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Additivemanufacturinglaserpowder()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_additive_manufacturing_laser_powder_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_additive_manufacturing_laser_powder_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_additive_manufacturing_laser_powder_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_additive_manufacturing_laser_powder_empty():
    extractor = FatigueCycleAccumulationExtractor_Additivemanufacturinglaserpowder()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
