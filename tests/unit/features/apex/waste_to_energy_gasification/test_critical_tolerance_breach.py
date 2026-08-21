# Unit Test for CriticalToleranceBreachExtractor_Wastetoenergygasification (Plasma Gasification Waste-to-Energy).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.waste_to_energy_gasification.critical_tolerance_breach import CriticalToleranceBreachExtractor_Wastetoenergygasification
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_waste_to_energy_gasification_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Wastetoenergygasification()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_waste_to_energy_gasification_signal" in res.columns
    assert f"critical_tolerance_breach_waste_to_energy_gasification_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_waste_to_energy_gasification_signal"].isnull().any()

def test_critical_tolerance_breach_waste_to_energy_gasification_empty():
    extractor = CriticalToleranceBreachExtractor_Wastetoenergygasification()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
