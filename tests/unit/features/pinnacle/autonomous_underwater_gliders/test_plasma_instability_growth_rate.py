# Unit Test for PlasmaInstabilityGrowthExtractor_Autonomousunderwatergliders (Oceanographic Autonomous Underwater Gliders).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.autonomous_underwater_gliders.plasma_instability_growth_rate import PlasmaInstabilityGrowthExtractor_Autonomousunderwatergliders
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_plasma_instability_growth_rate_autonomous_underwater_gliders_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PlasmaInstabilityGrowthExtractor_Autonomousunderwatergliders()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"plasma_instability_growth_rate_autonomous_underwater_gliders_signal" in res.columns
    assert f"plasma_instability_growth_rate_autonomous_underwater_gliders_risk_score" in res.columns
    assert not res[f"plasma_instability_growth_rate_autonomous_underwater_gliders_signal"].isnull().any()

def test_plasma_instability_growth_rate_autonomous_underwater_gliders_empty():
    extractor = PlasmaInstabilityGrowthExtractor_Autonomousunderwatergliders()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
