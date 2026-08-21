# Unit test for ExportVolumeSpikeExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.export_volume_spike import ExportVolumeSpikeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_export_volume_spike_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = ExportVolumeSpikeExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
