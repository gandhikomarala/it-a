# Unit Test for CriticalToleranceBreachExtractor_Mininghaultrucks (Autonomous Mining Haul Truck Fleets).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.mining_haul_trucks.critical_tolerance_breach import CriticalToleranceBreachExtractor_Mininghaultrucks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_mining_haul_trucks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Mininghaultrucks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_mining_haul_trucks_signal" in res.columns
    assert f"critical_tolerance_breach_mining_haul_trucks_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_mining_haul_trucks_signal"].isnull().any()

def test_critical_tolerance_breach_mining_haul_trucks_empty():
    extractor = CriticalToleranceBreachExtractor_Mininghaultrucks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
