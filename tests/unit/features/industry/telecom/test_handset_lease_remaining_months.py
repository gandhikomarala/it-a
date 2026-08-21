# Unit Test for HandsetLeaseRemainingMonths (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.handset_lease_remaining_months import HandsetLeaseRemainingMonths
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_handset_lease_remaining_months_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HandsetLeaseRemainingMonths()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"handset_lease_remaining_months_signal" in res.columns
    assert f"handset_lease_remaining_months_risk_index" in res.columns
    assert not res[f"handset_lease_remaining_months_signal"].isnull().any()

def test_handset_lease_remaining_months_empty_handling():
    extractor = HandsetLeaseRemainingMonths()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
