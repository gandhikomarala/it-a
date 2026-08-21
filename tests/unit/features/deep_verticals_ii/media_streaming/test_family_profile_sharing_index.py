# Comprehensive Unit Test for FamilyProfileSharingIndexExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.family_profile_sharing_index import FamilyProfileSharingIndexExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_family_profile_sharing_index_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FamilyProfileSharingIndexExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"family_profile_sharing_index_signal" in res.columns
    assert f"family_profile_sharing_index_risk_score" in res.columns
    assert not res[f"family_profile_sharing_index_signal"].isnull().any()

def test_family_profile_sharing_index_empty_handling():
    extractor = FamilyProfileSharingIndexExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
