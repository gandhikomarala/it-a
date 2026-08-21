# Comprehensive Unit Test for BLTelexReleaseDelayExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.bill_of_lading_telex_release_delay import BLTelexReleaseDelayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_bill_of_lading_telex_release_delay_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BLTelexReleaseDelayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"bill_of_lading_telex_release_delay_signal" in res.columns
    assert f"bill_of_lading_telex_release_delay_risk_score" in res.columns
    assert not res[f"bill_of_lading_telex_release_delay_signal"].isnull().any()

def test_bill_of_lading_telex_release_delay_empty_handling():
    extractor = BLTelexReleaseDelayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
