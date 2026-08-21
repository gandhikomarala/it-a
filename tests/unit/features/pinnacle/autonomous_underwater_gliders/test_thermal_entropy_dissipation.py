# Unit Test for ThermalEntropyDissipationExtractor_Autonomousunderwatergliders (Oceanographic Autonomous Underwater Gliders).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.autonomous_underwater_gliders.thermal_entropy_dissipation import ThermalEntropyDissipationExtractor_Autonomousunderwatergliders
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_thermal_entropy_dissipation_autonomous_underwater_gliders_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ThermalEntropyDissipationExtractor_Autonomousunderwatergliders()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"thermal_entropy_dissipation_autonomous_underwater_gliders_signal" in res.columns
    assert f"thermal_entropy_dissipation_autonomous_underwater_gliders_risk_score" in res.columns
    assert not res[f"thermal_entropy_dissipation_autonomous_underwater_gliders_signal"].isnull().any()

def test_thermal_entropy_dissipation_autonomous_underwater_gliders_empty():
    extractor = ThermalEntropyDissipationExtractor_Autonomousunderwatergliders()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
