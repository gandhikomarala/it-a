# Unit Test for MultivariateOutlierValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.multivariate_outlier_rule import MultivariateOutlierValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_multivariate_outlier_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = MultivariateOutlierValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
