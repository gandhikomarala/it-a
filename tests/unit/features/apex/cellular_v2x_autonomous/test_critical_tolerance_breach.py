# Unit Test for CriticalToleranceBreachExtractor_Cellularv2Xautonomous (Cellular V2X Connected Vehicle Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.cellular_v2x_autonomous.critical_tolerance_breach import CriticalToleranceBreachExtractor_Cellularv2Xautonomous
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_cellular_v2x_autonomous_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Cellularv2Xautonomous()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_cellular_v2x_autonomous_signal" in res.columns
    assert f"critical_tolerance_breach_cellular_v2x_autonomous_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_cellular_v2x_autonomous_signal"].isnull().any()

def test_critical_tolerance_breach_cellular_v2x_autonomous_empty():
    extractor = CriticalToleranceBreachExtractor_Cellularv2Xautonomous()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
