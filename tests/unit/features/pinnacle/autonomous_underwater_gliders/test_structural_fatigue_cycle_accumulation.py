# Unit Test for FatigueCycleAccumulationExtractor_Autonomousunderwatergliders (Oceanographic Autonomous Underwater Gliders).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.autonomous_underwater_gliders.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Autonomousunderwatergliders
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_autonomous_underwater_gliders_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Autonomousunderwatergliders()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_autonomous_underwater_gliders_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_autonomous_underwater_gliders_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_autonomous_underwater_gliders_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_autonomous_underwater_gliders_empty():
    extractor = FatigueCycleAccumulationExtractor_Autonomousunderwatergliders()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
