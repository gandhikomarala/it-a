# Unit Test for SystemEntropyScoreExtractor_Bioprintingtissuescaffolds (3D Extrusion Bioprinting Organ Scaffolds).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.bioprinting_tissue_scaffolds.system_entropy_score import SystemEntropyScoreExtractor_Bioprintingtissuescaffolds
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_bioprinting_tissue_scaffolds_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Bioprintingtissuescaffolds()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_bioprinting_tissue_scaffolds_signal" in res.columns
    assert f"system_entropy_score_bioprinting_tissue_scaffolds_risk_score" in res.columns
    assert not res[f"system_entropy_score_bioprinting_tissue_scaffolds_signal"].isnull().any()

def test_system_entropy_score_bioprinting_tissue_scaffolds_empty():
    extractor = SystemEntropyScoreExtractor_Bioprintingtissuescaffolds()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
