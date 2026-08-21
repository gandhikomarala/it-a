# Unit Test for RetentionHealthIndexExtractor_Pcbsmtassembly (Surface Mount PCB Assembly Quality).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.pcb_smt_assembly.retention_health_index import RetentionHealthIndexExtractor_Pcbsmtassembly
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_pcb_smt_assembly_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Pcbsmtassembly()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_pcb_smt_assembly_signal" in res.columns
    assert f"retention_health_index_pcb_smt_assembly_risk_score" in res.columns
    assert not res[f"retention_health_index_pcb_smt_assembly_signal"].isnull().any()

def test_retention_health_index_pcb_smt_assembly_empty():
    extractor = RetentionHealthIndexExtractor_Pcbsmtassembly()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
