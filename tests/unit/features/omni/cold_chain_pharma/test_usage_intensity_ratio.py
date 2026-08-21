# Unit Test for UsageIntensityRatioExtractor_Coldchainpharma (Cold Chain Biopharma Logistics).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cold_chain_pharma.usage_intensity_ratio import UsageIntensityRatioExtractor_Coldchainpharma
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_cold_chain_pharma_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Coldchainpharma()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_cold_chain_pharma_signal" in res.columns
    assert f"usage_intensity_ratio_cold_chain_pharma_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_cold_chain_pharma_signal"].isnull().any()

def test_usage_intensity_ratio_cold_chain_pharma_empty():
    extractor = UsageIntensityRatioExtractor_Coldchainpharma()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
