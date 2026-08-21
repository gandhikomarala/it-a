# Unit Test for LicenseRenewalGraceDays (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.license_renewal_grace_days import LicenseRenewalGraceDays
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_license_renewal_grace_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LicenseRenewalGraceDays()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"license_renewal_grace_days_signal" in res.columns
    assert f"license_renewal_grace_days_risk_index" in res.columns
    assert not res[f"license_renewal_grace_days_signal"].isnull().any()

def test_license_renewal_grace_days_empty_handling():
    extractor = LicenseRenewalGraceDays()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
