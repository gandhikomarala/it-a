# Unit Test for HeatLeakRateExtractor_Precisionagriculturehyperspectral (Drone SWIR Crop Water Stress Imaging).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.precision_agriculture_hyperspectral.cryogenic_boiloff_heat_leak_rate import HeatLeakRateExtractor_Precisionagriculturehyperspectral
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cryogenic_boiloff_heat_leak_rate_precision_agriculture_hyperspectral_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HeatLeakRateExtractor_Precisionagriculturehyperspectral()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cryogenic_boiloff_heat_leak_rate_precision_agriculture_hyperspectral_signal" in res.columns
    assert f"cryogenic_boiloff_heat_leak_rate_precision_agriculture_hyperspectral_risk_score" in res.columns
    assert not res[f"cryogenic_boiloff_heat_leak_rate_precision_agriculture_hyperspectral_signal"].isnull().any()

def test_cryogenic_boiloff_heat_leak_rate_precision_agriculture_hyperspectral_empty():
    extractor = HeatLeakRateExtractor_Precisionagriculturehyperspectral()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
