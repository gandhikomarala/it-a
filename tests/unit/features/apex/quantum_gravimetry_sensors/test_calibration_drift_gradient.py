# Unit Test for CalibrationDriftGradientExtractor_Quantumgravimetrysensors (Cold Atom Quantum Gravimetry Sensors).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.quantum_gravimetry_sensors.calibration_drift_gradient import CalibrationDriftGradientExtractor_Quantumgravimetrysensors
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_quantum_gravimetry_sensors_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Quantumgravimetrysensors()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_quantum_gravimetry_sensors_signal" in res.columns
    assert f"calibration_drift_gradient_quantum_gravimetry_sensors_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_quantum_gravimetry_sensors_signal"].isnull().any()

def test_calibration_drift_gradient_quantum_gravimetry_sensors_empty():
    extractor = CalibrationDriftGradientExtractor_Quantumgravimetrysensors()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
