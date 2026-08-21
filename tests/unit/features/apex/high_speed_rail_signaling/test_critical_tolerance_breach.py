# Unit Test for CriticalToleranceBreachExtractor_Highspeedrailsignaling (High-Speed Rail Positive Train Control).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.high_speed_rail_signaling.critical_tolerance_breach import CriticalToleranceBreachExtractor_Highspeedrailsignaling
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_high_speed_rail_signaling_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Highspeedrailsignaling()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_high_speed_rail_signaling_signal" in res.columns
    assert f"critical_tolerance_breach_high_speed_rail_signaling_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_high_speed_rail_signaling_signal"].isnull().any()

def test_critical_tolerance_breach_high_speed_rail_signaling_empty():
    extractor = CriticalToleranceBreachExtractor_Highspeedrailsignaling()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
