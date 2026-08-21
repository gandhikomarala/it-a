# Unit Test for CriticalCurrentMarginExtractor_Hyperspectralmineralexploration (Airborne Hyperspectral Mineral Mapping).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.hyperspectral_mineral_exploration.superconducting_critical_current_margin import CriticalCurrentMarginExtractor_Hyperspectralmineralexploration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_superconducting_critical_current_margin_hyperspectral_mineral_exploration_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalCurrentMarginExtractor_Hyperspectralmineralexploration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"superconducting_critical_current_margin_hyperspectral_mineral_exploration_signal" in res.columns
    assert f"superconducting_critical_current_margin_hyperspectral_mineral_exploration_risk_score" in res.columns
    assert not res[f"superconducting_critical_current_margin_hyperspectral_mineral_exploration_signal"].isnull().any()

def test_superconducting_critical_current_margin_hyperspectral_mineral_exploration_empty():
    extractor = CriticalCurrentMarginExtractor_Hyperspectralmineralexploration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
