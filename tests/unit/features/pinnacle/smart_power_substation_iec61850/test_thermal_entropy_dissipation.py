# Unit Test for ThermalEntropyDissipationExtractor_Smartpowersubstationiec61850 (Digital Substation IEC 61850 Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.smart_power_substation_iec61850.thermal_entropy_dissipation import ThermalEntropyDissipationExtractor_Smartpowersubstationiec61850
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_thermal_entropy_dissipation_smart_power_substation_iec61850_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ThermalEntropyDissipationExtractor_Smartpowersubstationiec61850()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"thermal_entropy_dissipation_smart_power_substation_iec61850_signal" in res.columns
    assert f"thermal_entropy_dissipation_smart_power_substation_iec61850_risk_score" in res.columns
    assert not res[f"thermal_entropy_dissipation_smart_power_substation_iec61850_signal"].isnull().any()

def test_thermal_entropy_dissipation_smart_power_substation_iec61850_empty():
    extractor = ThermalEntropyDissipationExtractor_Smartpowersubstationiec61850()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
