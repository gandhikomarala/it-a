# Comprehensive Unit Test for CertificateDownloadCadenceExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.certificate_download_cadence import CertificateDownloadCadenceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_certificate_download_cadence_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CertificateDownloadCadenceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"certificate_download_cadence_signal" in res.columns
    assert f"certificate_download_cadence_risk_score" in res.columns
    assert not res[f"certificate_download_cadence_signal"].isnull().any()

def test_certificate_download_cadence_empty():
    extractor = CertificateDownloadCadenceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
