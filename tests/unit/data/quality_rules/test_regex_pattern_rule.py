# Unit Test for RegexPatternValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.regex_pattern_rule import RegexPatternValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_regex_pattern_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = RegexPatternValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
