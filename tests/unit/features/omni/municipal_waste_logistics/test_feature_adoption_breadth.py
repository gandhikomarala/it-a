# Unit Test for FeatureAdoptionBreadthExtractor_Municipalwastelogistics (Municipal Smart Waste Routing).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.municipal_waste_logistics.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Municipalwastelogistics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_municipal_waste_logistics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Municipalwastelogistics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_municipal_waste_logistics_signal" in res.columns
    assert f"feature_adoption_breadth_municipal_waste_logistics_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_municipal_waste_logistics_signal"].isnull().any()

def test_feature_adoption_breadth_municipal_waste_logistics_empty():
    extractor = FeatureAdoptionBreadthExtractor_Municipalwastelogistics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
