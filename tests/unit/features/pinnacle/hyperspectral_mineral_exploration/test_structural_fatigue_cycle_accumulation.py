# Unit Test for FatigueCycleAccumulationExtractor_Hyperspectralmineralexploration (Airborne Hyperspectral Mineral Mapping).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.hyperspectral_mineral_exploration.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Hyperspectralmineralexploration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_hyperspectral_mineral_exploration_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Hyperspectralmineralexploration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_hyperspectral_mineral_exploration_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_hyperspectral_mineral_exploration_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_hyperspectral_mineral_exploration_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_hyperspectral_mineral_exploration_empty():
    extractor = FatigueCycleAccumulationExtractor_Hyperspectralmineralexploration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
