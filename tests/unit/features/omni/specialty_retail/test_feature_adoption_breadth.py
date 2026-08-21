# Unit Test for FeatureAdoptionBreadthExtractor_Specialtyretail (Specialty Retail Omnichannel Inventory).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.specialty_retail.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Specialtyretail
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_specialty_retail_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Specialtyretail()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_specialty_retail_signal" in res.columns
    assert f"feature_adoption_breadth_specialty_retail_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_specialty_retail_signal"].isnull().any()

def test_feature_adoption_breadth_specialty_retail_empty():
    extractor = FeatureAdoptionBreadthExtractor_Specialtyretail()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
