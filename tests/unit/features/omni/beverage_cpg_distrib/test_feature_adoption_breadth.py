# Unit Test for FeatureAdoptionBreadthExtractor_Beveragecpgdistrib (Beverage CPG Direct Store Delivery).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.beverage_cpg_distrib.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Beveragecpgdistrib
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_beverage_cpg_distrib_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Beveragecpgdistrib()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_beverage_cpg_distrib_signal" in res.columns
    assert f"feature_adoption_breadth_beverage_cpg_distrib_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_beverage_cpg_distrib_signal"].isnull().any()

def test_feature_adoption_breadth_beverage_cpg_distrib_empty():
    extractor = FeatureAdoptionBreadthExtractor_Beveragecpgdistrib()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
