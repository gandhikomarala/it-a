# Unit Test for FeatureAdoptionBreadthExtractor_Cruiselineshospitality (Cruise Line Passenger Lifetime Value).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cruise_lines_hospitality.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Cruiselineshospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_cruise_lines_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Cruiselineshospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_cruise_lines_hospitality_signal" in res.columns
    assert f"feature_adoption_breadth_cruise_lines_hospitality_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_cruise_lines_hospitality_signal"].isnull().any()

def test_feature_adoption_breadth_cruise_lines_hospitality_empty():
    extractor = FeatureAdoptionBreadthExtractor_Cruiselineshospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
