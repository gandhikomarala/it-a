# Unit Test for CriticalToleranceBreachExtractor_Cryogenicliquidhydrogen (Cryogenic Liquid Hydrogen Transport & Storage).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.cryogenic_liquid_hydrogen.critical_tolerance_breach import CriticalToleranceBreachExtractor_Cryogenicliquidhydrogen
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_cryogenic_liquid_hydrogen_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Cryogenicliquidhydrogen()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_cryogenic_liquid_hydrogen_signal" in res.columns
    assert f"critical_tolerance_breach_cryogenic_liquid_hydrogen_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_cryogenic_liquid_hydrogen_signal"].isnull().any()

def test_critical_tolerance_breach_cryogenic_liquid_hydrogen_empty():
    extractor = CriticalToleranceBreachExtractor_Cryogenicliquidhydrogen()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
