# Unit Test for TransientVoltageOvershootExtractor_Neuromorphicretinavision (Event-Based Neuromorphic Silicon Retina).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.neuromorphic_retina_vision.transient_voltage_overshoot import TransientVoltageOvershootExtractor_Neuromorphicretinavision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_transient_voltage_overshoot_neuromorphic_retina_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TransientVoltageOvershootExtractor_Neuromorphicretinavision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"transient_voltage_overshoot_neuromorphic_retina_vision_signal" in res.columns
    assert f"transient_voltage_overshoot_neuromorphic_retina_vision_risk_score" in res.columns
    assert not res[f"transient_voltage_overshoot_neuromorphic_retina_vision_signal"].isnull().any()

def test_transient_voltage_overshoot_neuromorphic_retina_vision_empty():
    extractor = TransientVoltageOvershootExtractor_Neuromorphicretinavision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
