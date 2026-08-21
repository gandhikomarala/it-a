# Unit Test for FeatureAdoptionBreadthExtractor_Waterutilityami (Municipal Smart Water AMI Network).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.water_utility_ami.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Waterutilityami
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_water_utility_ami_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Waterutilityami()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_water_utility_ami_signal" in res.columns
    assert f"feature_adoption_breadth_water_utility_ami_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_water_utility_ami_signal"].isnull().any()

def test_feature_adoption_breadth_water_utility_ami_empty():
    extractor = FeatureAdoptionBreadthExtractor_Waterutilityami()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
