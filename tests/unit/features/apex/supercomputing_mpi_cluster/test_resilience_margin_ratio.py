# Unit Test for ResilienceMarginRatioExtractor_Supercomputingmpicluster (Exascale Supercomputing MPI Interconnects).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.supercomputing_mpi_cluster.resilience_margin_ratio import ResilienceMarginRatioExtractor_Supercomputingmpicluster
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_supercomputing_mpi_cluster_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Supercomputingmpicluster()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_supercomputing_mpi_cluster_signal" in res.columns
    assert f"resilience_margin_ratio_supercomputing_mpi_cluster_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_supercomputing_mpi_cluster_signal"].isnull().any()

def test_resilience_margin_ratio_supercomputing_mpi_cluster_empty():
    extractor = ResilienceMarginRatioExtractor_Supercomputingmpicluster()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
