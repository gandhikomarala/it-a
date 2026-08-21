# Unit Test for SAMLSSOConfigured (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.saml_sso_configured import SAMLSSOConfigured
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_saml_sso_configured_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SAMLSSOConfigured()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"saml_sso_configured_signal" in res.columns
    assert f"saml_sso_configured_risk_index" in res.columns
    assert not res[f"saml_sso_configured_signal"].isnull().any()

def test_saml_sso_configured_empty_handling():
    extractor = SAMLSSOConfigured()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
