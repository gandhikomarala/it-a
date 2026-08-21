# Comprehensive Unit Test for EmptyContainerRepositioningExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.empty_container_repositioning_cost import EmptyContainerRepositioningExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_empty_container_repositioning_cost_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EmptyContainerRepositioningExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"empty_container_repositioning_cost_signal" in res.columns
    assert f"empty_container_repositioning_cost_risk_score" in res.columns
    assert not res[f"empty_container_repositioning_cost_signal"].isnull().any()

def test_empty_container_repositioning_cost_empty_handling():
    extractor = EmptyContainerRepositioningExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
