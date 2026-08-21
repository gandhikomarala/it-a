# Unit Test for SatisfactionDriftDeltaExtractor_Biotechgenomics (Biotech & Next-Gen Sequencing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.biotech_genomics.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Biotechgenomics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_biotech_genomics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Biotechgenomics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_biotech_genomics_signal" in res.columns
    assert f"satisfaction_drift_delta_biotech_genomics_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_biotech_genomics_signal"].isnull().any()

def test_satisfaction_drift_delta_biotech_genomics_empty():
    extractor = SatisfactionDriftDeltaExtractor_Biotechgenomics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
