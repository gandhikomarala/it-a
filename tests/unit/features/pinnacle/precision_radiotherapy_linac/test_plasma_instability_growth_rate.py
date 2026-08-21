# Unit Test for PlasmaInstabilityGrowthExtractor_Precisionradiotherapylinac (Linear Accelerator Medical Radiotherapy).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.precision_radiotherapy_linac.plasma_instability_growth_rate import PlasmaInstabilityGrowthExtractor_Precisionradiotherapylinac
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_plasma_instability_growth_rate_precision_radiotherapy_linac_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PlasmaInstabilityGrowthExtractor_Precisionradiotherapylinac()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"plasma_instability_growth_rate_precision_radiotherapy_linac_signal" in res.columns
    assert f"plasma_instability_growth_rate_precision_radiotherapy_linac_risk_score" in res.columns
    assert not res[f"plasma_instability_growth_rate_precision_radiotherapy_linac_signal"].isnull().any()

def test_plasma_instability_growth_rate_precision_radiotherapy_linac_empty():
    extractor = PlasmaInstabilityGrowthExtractor_Precisionradiotherapylinac()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
