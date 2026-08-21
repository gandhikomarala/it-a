# Unit Test for FuelSurchargeDisputeRateExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.fuel_surcharge_dispute_rate import FuelSurchargeDisputeRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_fuel_surcharge_dispute_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FuelSurchargeDisputeRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"fuel_surcharge_dispute_rate_signal" in res.columns
    assert f"fuel_surcharge_dispute_rate_risk_score" in res.columns
    assert not res[f"fuel_surcharge_dispute_rate_signal"].isnull().any()

def test_fuel_surcharge_dispute_rate_empty_dataframe():
    extractor = FuelSurchargeDisputeRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
