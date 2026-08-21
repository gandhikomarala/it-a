# Unit Test for CriticalToleranceBreachExtractor_Supercomputingmpicluster (Exascale Supercomputing MPI Interconnects).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.supercomputing_mpi_cluster.critical_tolerance_breach import CriticalToleranceBreachExtractor_Supercomputingmpicluster
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_supercomputing_mpi_cluster_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Supercomputingmpicluster()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_supercomputing_mpi_cluster_signal" in res.columns
    assert f"critical_tolerance_breach_supercomputing_mpi_cluster_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_supercomputing_mpi_cluster_signal"].isnull().any()

def test_critical_tolerance_breach_supercomputing_mpi_cluster_empty():
    extractor = CriticalToleranceBreachExtractor_Supercomputingmpicluster()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
