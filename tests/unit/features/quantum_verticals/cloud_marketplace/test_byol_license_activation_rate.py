# Comprehensive Unit Test for BYOLLicenseActivationRateExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.byol_license_activation_rate import BYOLLicenseActivationRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_byol_license_activation_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BYOLLicenseActivationRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"byol_license_activation_rate_signal" in res.columns
    assert f"byol_license_activation_rate_risk_score" in res.columns
    assert not res[f"byol_license_activation_rate_signal"].isnull().any()

def test_byol_license_activation_rate_empty_handling():
    extractor = BYOLLicenseActivationRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
