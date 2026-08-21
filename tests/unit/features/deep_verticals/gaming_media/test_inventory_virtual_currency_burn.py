# Comprehensive Unit Test for VirtualCurrencyBurnExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.inventory_virtual_currency_burn import VirtualCurrencyBurnExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_inventory_virtual_currency_burn_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VirtualCurrencyBurnExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"inventory_virtual_currency_burn_signal" in res.columns
    assert f"inventory_virtual_currency_burn_risk_score" in res.columns
    assert not res[f"inventory_virtual_currency_burn_signal"].isnull().any()

def test_inventory_virtual_currency_burn_empty():
    extractor = VirtualCurrencyBurnExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
