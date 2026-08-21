# Unit Test for ThermalEntropyDissipationExtractor_Hyperspectralmineralexploration (Airborne Hyperspectral Mineral Mapping).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.hyperspectral_mineral_exploration.thermal_entropy_dissipation import ThermalEntropyDissipationExtractor_Hyperspectralmineralexploration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_thermal_entropy_dissipation_hyperspectral_mineral_exploration_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ThermalEntropyDissipationExtractor_Hyperspectralmineralexploration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"thermal_entropy_dissipation_hyperspectral_mineral_exploration_signal" in res.columns
    assert f"thermal_entropy_dissipation_hyperspectral_mineral_exploration_risk_score" in res.columns
    assert not res[f"thermal_entropy_dissipation_hyperspectral_mineral_exploration_signal"].isnull().any()

def test_thermal_entropy_dissipation_hyperspectral_mineral_exploration_empty():
    extractor = ThermalEntropyDissipationExtractor_Hyperspectralmineralexploration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
