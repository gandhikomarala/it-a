# Comprehensive Unit Test for ResellerMarginDecayExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.reseller_partner_margin_decay import ResellerMarginDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_reseller_partner_margin_decay_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResellerMarginDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"reseller_partner_margin_decay_signal" in res.columns
    assert f"reseller_partner_margin_decay_risk_score" in res.columns
    assert not res[f"reseller_partner_margin_decay_signal"].isnull().any()

def test_reseller_partner_margin_decay_empty_handling():
    extractor = ResellerMarginDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
