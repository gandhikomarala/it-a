# Unit Test for LifecycleBurnRateExtractor_Synbiofermentation (Synthetic Biology & Bioreactor Fermentation).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.synbio_fermentation.lifecycle_burn_rate import LifecycleBurnRateExtractor_Synbiofermentation
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_synbio_fermentation_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Synbiofermentation()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_synbio_fermentation_signal" in res.columns
    assert f"lifecycle_burn_rate_synbio_fermentation_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_synbio_fermentation_signal"].isnull().any()

def test_lifecycle_burn_rate_synbio_fermentation_empty():
    extractor = LifecycleBurnRateExtractor_Synbiofermentation()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
