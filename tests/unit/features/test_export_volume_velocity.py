# Comprehensive Unit Test for ExportVolumeVelocityExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.export_volume_velocity import ExportVolumeVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_export_volume_velocity_instantiation():
    extractor = ExportVolumeVelocityExtractor()
    assert extractor.prefix == "export_volume_velocity"

def test_export_volume_velocity_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = ExportVolumeVelocityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("export_volume_velocity_")]
    assert len(expected_cols) > 0

def test_export_volume_velocity_transform_empty():
    extractor = ExportVolumeVelocityExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
