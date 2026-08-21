# Comprehensive Unit Test for CropInsuranceClaimFiledExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.crop_insurance_claim_filed import CropInsuranceClaimFiledExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_crop_insurance_claim_filed_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CropInsuranceClaimFiledExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"crop_insurance_claim_filed_signal" in res.columns
    assert f"crop_insurance_claim_filed_risk_score" in res.columns
    assert not res[f"crop_insurance_claim_filed_signal"].isnull().any()

def test_crop_insurance_claim_filed_empty_handling():
    extractor = CropInsuranceClaimFiledExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
