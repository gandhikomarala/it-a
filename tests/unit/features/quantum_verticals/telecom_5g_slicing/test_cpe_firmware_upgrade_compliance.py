# Comprehensive Unit Test for CPEFirmwareComplianceExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.cpe_firmware_upgrade_compliance import CPEFirmwareComplianceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cpe_firmware_upgrade_compliance_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CPEFirmwareComplianceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cpe_firmware_upgrade_compliance_signal" in res.columns
    assert f"cpe_firmware_upgrade_compliance_risk_score" in res.columns
    assert not res[f"cpe_firmware_upgrade_compliance_signal"].isnull().any()

def test_cpe_firmware_upgrade_compliance_empty_handling():
    extractor = CPEFirmwareComplianceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
