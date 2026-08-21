# Unit Test for FeatureAdoptionBreadthExtractor_Miningheavyindustry (Mining & Heavy Equipment IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.mining_heavy_industry.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Miningheavyindustry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_mining_heavy_industry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Miningheavyindustry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_mining_heavy_industry_signal" in res.columns
    assert f"feature_adoption_breadth_mining_heavy_industry_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_mining_heavy_industry_signal"].isnull().any()

def test_feature_adoption_breadth_mining_heavy_industry_empty():
    extractor = FeatureAdoptionBreadthExtractor_Miningheavyindustry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
