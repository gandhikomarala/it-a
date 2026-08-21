# Comprehensive Unit Test for DemurrageChargesExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.container_demurrage_detention_charges import DemurrageChargesExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_container_demurrage_detention_charges_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DemurrageChargesExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"container_demurrage_detention_charges_signal" in res.columns
    assert f"container_demurrage_detention_charges_risk_score" in res.columns
    assert not res[f"container_demurrage_detention_charges_signal"].isnull().any()

def test_container_demurrage_detention_charges_empty_handling():
    extractor = DemurrageChargesExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
