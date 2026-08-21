# Unit Test for HarmonicDistortionFactorExtractor_Directlithiumextractionbrine (Geothermal Brine Direct Lithium Extraction).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.direct_lithium_extraction_brine.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Directlithiumextractionbrine
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_direct_lithium_extraction_brine_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Directlithiumextractionbrine()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_direct_lithium_extraction_brine_signal" in res.columns
    assert f"harmonic_distortion_factor_direct_lithium_extraction_brine_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_direct_lithium_extraction_brine_signal"].isnull().any()

def test_harmonic_distortion_factor_direct_lithium_extraction_brine_empty():
    extractor = HarmonicDistortionFactorExtractor_Directlithiumextractionbrine()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
