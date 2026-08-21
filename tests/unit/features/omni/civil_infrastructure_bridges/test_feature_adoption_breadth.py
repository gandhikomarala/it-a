# Unit Test for FeatureAdoptionBreadthExtractor_Civilinfrastructurebridges (Civil Infrastructure & Bridge Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.civil_infrastructure_bridges.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Civilinfrastructurebridges
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_civil_infrastructure_bridges_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Civilinfrastructurebridges()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_civil_infrastructure_bridges_signal" in res.columns
    assert f"feature_adoption_breadth_civil_infrastructure_bridges_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_civil_infrastructure_bridges_signal"].isnull().any()

def test_feature_adoption_breadth_civil_infrastructure_bridges_empty():
    extractor = FeatureAdoptionBreadthExtractor_Civilinfrastructurebridges()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
