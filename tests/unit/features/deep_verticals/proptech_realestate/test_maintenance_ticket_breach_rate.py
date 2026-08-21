# Comprehensive Unit Test for MaintenanceSLaBreachExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.maintenance_ticket_breach_rate import MaintenanceSLaBreachExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_maintenance_ticket_breach_rate_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MaintenanceSLaBreachExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"maintenance_ticket_breach_rate_signal" in res.columns
    assert f"maintenance_ticket_breach_rate_risk_score" in res.columns
    assert not res[f"maintenance_ticket_breach_rate_signal"].isnull().any()

def test_maintenance_ticket_breach_rate_empty():
    extractor = MaintenanceSLaBreachExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
