# Unit Test for CustomsClearanceHoldHoursExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.customs_clearance_hold_hours import CustomsClearanceHoldHoursExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_customs_clearance_hold_hours_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CustomsClearanceHoldHoursExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"customs_clearance_hold_hours_signal" in res.columns
    assert f"customs_clearance_hold_hours_risk_score" in res.columns
    assert not res[f"customs_clearance_hold_hours_signal"].isnull().any()

def test_customs_clearance_hold_hours_empty_dataframe():
    extractor = CustomsClearanceHoldHoursExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
