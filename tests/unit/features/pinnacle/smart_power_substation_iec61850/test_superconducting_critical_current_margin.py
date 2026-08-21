# Unit Test for CriticalCurrentMarginExtractor_Smartpowersubstationiec61850 (Digital Substation IEC 61850 Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.smart_power_substation_iec61850.superconducting_critical_current_margin import CriticalCurrentMarginExtractor_Smartpowersubstationiec61850
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_superconducting_critical_current_margin_smart_power_substation_iec61850_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalCurrentMarginExtractor_Smartpowersubstationiec61850()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"superconducting_critical_current_margin_smart_power_substation_iec61850_signal" in res.columns
    assert f"superconducting_critical_current_margin_smart_power_substation_iec61850_risk_score" in res.columns
    assert not res[f"superconducting_critical_current_margin_smart_power_substation_iec61850_signal"].isnull().any()

def test_superconducting_critical_current_margin_smart_power_substation_iec61850_empty():
    extractor = CriticalCurrentMarginExtractor_Smartpowersubstationiec61850()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
