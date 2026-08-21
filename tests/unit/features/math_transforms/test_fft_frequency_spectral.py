# Comprehensive Unit Test for FFTFrequencySpectralTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.fft_frequency_spectral import FFTFrequencySpectralTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_fft_frequency_spectral_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = FFTFrequencySpectralTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"fft_frequency_spectral_transformed" in res.columns
    assert not res[f"fft_frequency_spectral_transformed"].isnull().any()

def test_fft_frequency_spectral_numerical_stability():
    transformer = FFTFrequencySpectralTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"fft_frequency_spectral_transformed"]).any()
