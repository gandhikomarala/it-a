# Unit test for UsageAnomalyZScoreExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.usage_anomaly_zscore import UsageAnomalyZScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_anomaly_zscore_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = UsageAnomalyZScoreExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
