# Unit Test for CryptoAssetLiquidationFlowExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.crypto_asset_liquidation_flow import CryptoAssetLiquidationFlowExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_crypto_asset_liquidation_flow_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CryptoAssetLiquidationFlowExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"crypto_asset_liquidation_flow_signal" in res.columns
    assert f"crypto_asset_liquidation_flow_risk_score" in res.columns
    assert not res[f"crypto_asset_liquidation_flow_signal"].isnull().any()

def test_crypto_asset_liquidation_flow_empty_dataframe():
    extractor = CryptoAssetLiquidationFlowExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
