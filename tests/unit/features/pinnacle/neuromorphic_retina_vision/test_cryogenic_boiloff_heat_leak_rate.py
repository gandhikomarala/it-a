# Unit Test for HeatLeakRateExtractor_Neuromorphicretinavision (Event-Based Neuromorphic Silicon Retina).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.neuromorphic_retina_vision.cryogenic_boiloff_heat_leak_rate import HeatLeakRateExtractor_Neuromorphicretinavision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cryogenic_boiloff_heat_leak_rate_neuromorphic_retina_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HeatLeakRateExtractor_Neuromorphicretinavision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cryogenic_boiloff_heat_leak_rate_neuromorphic_retina_vision_signal" in res.columns
    assert f"cryogenic_boiloff_heat_leak_rate_neuromorphic_retina_vision_risk_score" in res.columns
    assert not res[f"cryogenic_boiloff_heat_leak_rate_neuromorphic_retina_vision_signal"].isnull().any()

def test_cryogenic_boiloff_heat_leak_rate_neuromorphic_retina_vision_empty():
    extractor = HeatLeakRateExtractor_Neuromorphicretinavision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
