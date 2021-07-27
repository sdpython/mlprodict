"""
@brief      test log(time=3s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.grammar_sklearn.grammar.api_extension import AutoType


class TestGrammarSklearnCov(ExtTestCase):

    def test_auto_type(self):
        at = AutoType()
        self.assertRaise(lambda: at.format_value(3), NotImplementedError)
        at._format_value_json = lambda v: str(v)  # pylint: disable=W0212
        self.assertRaise(lambda: at.format_value(3), TypeError)


if __name__ == "__main__":
    unittest.main()
