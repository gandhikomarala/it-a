# Comprehensive Unit Test for PropellantDepletionVelocityExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.propellant_mass_depletion_velocity import PropellantDepletionVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_propellant_mass_depletion_velocity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PropellantDepletionVelocityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"propellant_mass_depletion_velocity_signal" in res.columns
    assert f"propellant_mass_depletion_velocity_risk_score" in res.columns
    assert not res[f"propellant_mass_depletion_velocity_signal"].isnull().any()

def test_propellant_mass_depletion_velocity_empty_handling():
    extractor = PropellantDepletionVelocityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
