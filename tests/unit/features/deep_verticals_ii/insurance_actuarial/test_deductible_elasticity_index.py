# Comprehensive Unit Test for DeductibleElasticityIndexExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.deductible_elasticity_index import DeductibleElasticityIndexExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_deductible_elasticity_index_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DeductibleElasticityIndexExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"deductible_elasticity_index_signal" in res.columns
    assert f"deductible_elasticity_index_risk_score" in res.columns
    assert not res[f"deductible_elasticity_index_signal"].isnull().any()

def test_deductible_elasticity_index_empty_handling():
    extractor = DeductibleElasticityIndexExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
