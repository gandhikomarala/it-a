# Comprehensive Unit Test for MarketplaceTaxExemptionExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.marketplace_tax_exemption_status import MarketplaceTaxExemptionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_marketplace_tax_exemption_status_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MarketplaceTaxExemptionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"marketplace_tax_exemption_status_signal" in res.columns
    assert f"marketplace_tax_exemption_status_risk_score" in res.columns
    assert not res[f"marketplace_tax_exemption_status_signal"].isnull().any()

def test_marketplace_tax_exemption_status_empty_handling():
    extractor = MarketplaceTaxExemptionExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
