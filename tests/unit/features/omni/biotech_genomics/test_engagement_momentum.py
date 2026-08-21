# Unit Test for EngagementMomentumExtractor_Biotechgenomics (Biotech & Next-Gen Sequencing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.biotech_genomics.engagement_momentum import EngagementMomentumExtractor_Biotechgenomics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_biotech_genomics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Biotechgenomics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_biotech_genomics_signal" in res.columns
    assert f"engagement_momentum_biotech_genomics_risk_score" in res.columns
    assert not res[f"engagement_momentum_biotech_genomics_signal"].isnull().any()

def test_engagement_momentum_biotech_genomics_empty():
    extractor = EngagementMomentumExtractor_Biotechgenomics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
