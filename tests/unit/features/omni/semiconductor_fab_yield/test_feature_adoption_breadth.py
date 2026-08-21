# Unit Test for FeatureAdoptionBreadthExtractor_Semiconductorfabyield (Semiconductor 3nm Wafer Fab Yield).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.semiconductor_fab_yield.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Semiconductorfabyield
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_semiconductor_fab_yield_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Semiconductorfabyield()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_semiconductor_fab_yield_signal" in res.columns
    assert f"feature_adoption_breadth_semiconductor_fab_yield_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_semiconductor_fab_yield_signal"].isnull().any()

def test_feature_adoption_breadth_semiconductor_fab_yield_empty():
    extractor = FeatureAdoptionBreadthExtractor_Semiconductorfabyield()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
