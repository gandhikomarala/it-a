# Comprehensive Unit Test for PropertyTaxDisputesExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.property_tax_assessment_disputes import PropertyTaxDisputesExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_property_tax_assessment_disputes_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PropertyTaxDisputesExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"property_tax_assessment_disputes_signal" in res.columns
    assert f"property_tax_assessment_disputes_risk_score" in res.columns
    assert not res[f"property_tax_assessment_disputes_signal"].isnull().any()

def test_property_tax_assessment_disputes_empty_handling():
    extractor = PropertyTaxDisputesExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
