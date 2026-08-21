# Comprehensive Unit Test for BiomarkerSampleRejectionExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.biomarker_assay_sample_rejections import BiomarkerSampleRejectionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_biomarker_assay_sample_rejections_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BiomarkerSampleRejectionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"biomarker_assay_sample_rejections_signal" in res.columns
    assert f"biomarker_assay_sample_rejections_risk_score" in res.columns
    assert not res[f"biomarker_assay_sample_rejections_signal"].isnull().any()

def test_biomarker_assay_sample_rejections_empty_handling():
    extractor = BiomarkerSampleRejectionExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
