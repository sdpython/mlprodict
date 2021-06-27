"""
@brief      test log(time=3s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.einsum import decompose_einsum_equation


class TestEinsumBug(ExtTestCase):

    def test_abbba(self):
        res = decompose_einsum_equation(
            "ab,b->ba", strategy='numpy', clean=True)
        self.assertNotEmpty(res)

    def test__pprint_forward(self):
        res = decompose_einsum_equation(
            "ab,b->ba", strategy='numpy', clean=True)
        pf = res._pprint_forward()  # pylint: disable=W0212
        spl = pf.split("<- id")
        self.assertEqual(len(spl), 4)


if __name__ == "__main__":
    unittest.main()
