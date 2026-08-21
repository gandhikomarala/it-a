# Unit Test for EfficiencyDegradationPaceExtractor_Smartgridsynchrophasor (Smart Grid PMU Synchrophasor Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_grid_synchrophasor.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Smartgridsynchrophasor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_smart_grid_synchrophasor_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Smartgridsynchrophasor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_smart_grid_synchrophasor_signal" in res.columns
    assert f"efficiency_degradation_pace_smart_grid_synchrophasor_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_smart_grid_synchrophasor_signal"].isnull().any()

def test_efficiency_degradation_pace_smart_grid_synchrophasor_empty():
    extractor = EfficiencyDegradationPaceExtractor_Smartgridsynchrophasor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
