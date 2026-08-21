# Unit Test for InternationalRoamingActive (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.international_roaming_active import InternationalRoamingActive
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_international_roaming_active_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = InternationalRoamingActive()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"international_roaming_active_signal" in res.columns
    assert f"international_roaming_active_risk_index" in res.columns
    assert not res[f"international_roaming_active_signal"].isnull().any()

def test_international_roaming_active_empty_handling():
    extractor = InternationalRoamingActive()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
