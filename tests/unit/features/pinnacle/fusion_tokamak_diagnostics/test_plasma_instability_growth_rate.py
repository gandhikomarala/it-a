# Unit Test for PlasmaInstabilityGrowthExtractor_Fusiontokamakdiagnostics (Tokamak Fusion Energy Diagnostics).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.fusion_tokamak_diagnostics.plasma_instability_growth_rate import PlasmaInstabilityGrowthExtractor_Fusiontokamakdiagnostics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_plasma_instability_growth_rate_fusion_tokamak_diagnostics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PlasmaInstabilityGrowthExtractor_Fusiontokamakdiagnostics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"plasma_instability_growth_rate_fusion_tokamak_diagnostics_signal" in res.columns
    assert f"plasma_instability_growth_rate_fusion_tokamak_diagnostics_risk_score" in res.columns
    assert not res[f"plasma_instability_growth_rate_fusion_tokamak_diagnostics_signal"].isnull().any()

def test_plasma_instability_growth_rate_fusion_tokamak_diagnostics_empty():
    extractor = PlasmaInstabilityGrowthExtractor_Fusiontokamakdiagnostics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
