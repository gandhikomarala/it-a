# Unit Test for PlasmaInstabilityGrowthExtractor_Hyperspectralmineralexploration (Airborne Hyperspectral Mineral Mapping).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.hyperspectral_mineral_exploration.plasma_instability_growth_rate import PlasmaInstabilityGrowthExtractor_Hyperspectralmineralexploration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_plasma_instability_growth_rate_hyperspectral_mineral_exploration_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PlasmaInstabilityGrowthExtractor_Hyperspectralmineralexploration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"plasma_instability_growth_rate_hyperspectral_mineral_exploration_signal" in res.columns
    assert f"plasma_instability_growth_rate_hyperspectral_mineral_exploration_risk_score" in res.columns
    assert not res[f"plasma_instability_growth_rate_hyperspectral_mineral_exploration_signal"].isnull().any()

def test_plasma_instability_growth_rate_hyperspectral_mineral_exploration_empty():
    extractor = PlasmaInstabilityGrowthExtractor_Hyperspectralmineralexploration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
