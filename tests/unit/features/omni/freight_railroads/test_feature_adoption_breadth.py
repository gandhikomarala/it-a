# Unit Test for FeatureAdoptionBreadthExtractor_Freightrailroads (Class I Freight Railroad Logistics).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.freight_railroads.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Freightrailroads
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_freight_railroads_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Freightrailroads()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_freight_railroads_signal" in res.columns
    assert f"feature_adoption_breadth_freight_railroads_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_freight_railroads_signal"].isnull().any()

def test_feature_adoption_breadth_freight_railroads_empty():
    extractor = FeatureAdoptionBreadthExtractor_Freightrailroads()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
