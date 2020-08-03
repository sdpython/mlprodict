"""
@brief      test log(time=2s)
"""
import unittest
import pandas
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.cleaning import clean_error_msg


class TestCleaning(ExtTestCase):

    def test_cleaning(self):
        df = pandas.DataFrame(data=[
            dict(row="a", err="b"),
            dict(row="c", err="d")])
        res = clean_error_msg(df)
        self.assertIn("a", str(res))
        df = pandas.DataFrame(data=[
            dict(row="a", ERROR="b"),
            dict(row="b")])
        res = clean_error_msg(df)
        self.assertIn('NaN', str(res))


if __name__ == "__main__":
    unittest.main()
