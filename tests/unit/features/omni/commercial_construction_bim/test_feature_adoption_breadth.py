# Unit Test for FeatureAdoptionBreadthExtractor_Commercialconstructionbim (Commercial BIM Construction Tracking).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.commercial_construction_bim.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Commercialconstructionbim
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_commercial_construction_bim_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Commercialconstructionbim()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_commercial_construction_bim_signal" in res.columns
    assert f"feature_adoption_breadth_commercial_construction_bim_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_commercial_construction_bim_signal"].isnull().any()

def test_feature_adoption_breadth_commercial_construction_bim_empty():
    extractor = FeatureAdoptionBreadthExtractor_Commercialconstructionbim()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
