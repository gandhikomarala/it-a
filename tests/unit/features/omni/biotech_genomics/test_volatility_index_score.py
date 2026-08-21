# Unit Test for VolatilityIndexScoreExtractor_Biotechgenomics (Biotech & Next-Gen Sequencing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.biotech_genomics.volatility_index_score import VolatilityIndexScoreExtractor_Biotechgenomics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_biotech_genomics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Biotechgenomics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_biotech_genomics_signal" in res.columns
    assert f"volatility_index_score_biotech_genomics_risk_score" in res.columns
    assert not res[f"volatility_index_score_biotech_genomics_signal"].isnull().any()

def test_volatility_index_score_biotech_genomics_empty():
    extractor = VolatilityIndexScoreExtractor_Biotechgenomics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
