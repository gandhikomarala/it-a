# Unit Test for SSLCertificateExpirationDaysExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.ssl_certificate_expiration_days import SSLCertificateExpirationDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ssl_certificate_expiration_days_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SSLCertificateExpirationDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ssl_certificate_expiration_days_signal" in res.columns
    assert f"ssl_certificate_expiration_days_risk_score" in res.columns
    assert not res[f"ssl_certificate_expiration_days_signal"].isnull().any()

def test_ssl_certificate_expiration_days_empty_dataframe():
    extractor = SSLCertificateExpirationDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
