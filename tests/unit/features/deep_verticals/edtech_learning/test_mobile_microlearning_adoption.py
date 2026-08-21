# Comprehensive Unit Test for MicrolearningAdoptionExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.mobile_microlearning_adoption import MicrolearningAdoptionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_mobile_microlearning_adoption_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MicrolearningAdoptionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"mobile_microlearning_adoption_signal" in res.columns
    assert f"mobile_microlearning_adoption_risk_score" in res.columns
    assert not res[f"mobile_microlearning_adoption_signal"].isnull().any()

def test_mobile_microlearning_adoption_empty():
    extractor = MicrolearningAdoptionExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
