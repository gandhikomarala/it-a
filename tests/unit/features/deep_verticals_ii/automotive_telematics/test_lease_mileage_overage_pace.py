# Comprehensive Unit Test for LeaseMileageOveragePaceExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.lease_mileage_overage_pace import LeaseMileageOveragePaceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lease_mileage_overage_pace_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LeaseMileageOveragePaceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lease_mileage_overage_pace_signal" in res.columns
    assert f"lease_mileage_overage_pace_risk_score" in res.columns
    assert not res[f"lease_mileage_overage_pace_signal"].isnull().any()

def test_lease_mileage_overage_pace_empty_handling():
    extractor = LeaseMileageOveragePaceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
