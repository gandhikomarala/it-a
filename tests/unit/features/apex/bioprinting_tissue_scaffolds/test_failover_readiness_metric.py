# Unit Test for FailoverReadinessMetricExtractor_Bioprintingtissuescaffolds (3D Extrusion Bioprinting Organ Scaffolds).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.bioprinting_tissue_scaffolds.failover_readiness_metric import FailoverReadinessMetricExtractor_Bioprintingtissuescaffolds
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_bioprinting_tissue_scaffolds_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Bioprintingtissuescaffolds()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_bioprinting_tissue_scaffolds_signal" in res.columns
    assert f"failover_readiness_metric_bioprinting_tissue_scaffolds_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_bioprinting_tissue_scaffolds_signal"].isnull().any()

def test_failover_readiness_metric_bioprinting_tissue_scaffolds_empty():
    extractor = FailoverReadinessMetricExtractor_Bioprintingtissuescaffolds()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
