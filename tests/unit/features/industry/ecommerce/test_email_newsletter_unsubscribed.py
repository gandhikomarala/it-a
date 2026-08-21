# Unit Test for NewsletterUnsubscribed (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.email_newsletter_unsubscribed import NewsletterUnsubscribed
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_email_newsletter_unsubscribed_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NewsletterUnsubscribed()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"email_newsletter_unsubscribed_signal" in res.columns
    assert f"email_newsletter_unsubscribed_risk_index" in res.columns
    assert not res[f"email_newsletter_unsubscribed_signal"].isnull().any()

def test_email_newsletter_unsubscribed_empty_handling():
    extractor = NewsletterUnsubscribed()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
