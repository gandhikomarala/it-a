# Unit Test for FeatureAdoptionBreadthExtractor_Biotechgenomics (Biotech & Next-Gen Sequencing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.biotech_genomics.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Biotechgenomics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_biotech_genomics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Biotechgenomics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_biotech_genomics_signal" in res.columns
    assert f"feature_adoption_breadth_biotech_genomics_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_biotech_genomics_signal"].isnull().any()

def test_feature_adoption_breadth_biotech_genomics_empty():
    extractor = FeatureAdoptionBreadthExtractor_Biotechgenomics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
