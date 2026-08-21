# Unit Test for FeatureAdoptionBreadthExtractor_Defenseaerospace (Defense & Aerospace Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.defense_aerospace.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Defenseaerospace
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_defense_aerospace_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Defenseaerospace()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_defense_aerospace_signal" in res.columns
    assert f"feature_adoption_breadth_defense_aerospace_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_defense_aerospace_signal"].isnull().any()

def test_feature_adoption_breadth_defense_aerospace_empty():
    extractor = FeatureAdoptionBreadthExtractor_Defenseaerospace()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
