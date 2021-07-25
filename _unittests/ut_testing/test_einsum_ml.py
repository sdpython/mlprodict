"""
@brief      test log(time=3s)
"""
import unittest
from itertools import permutations
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.einsum.einsum_ml import (
    predict_transposition_cost, compute_transposition_features,
    _edit_distance)


class TestEinsumMl(ExtTestCase):

    def test_features(self):
        res = compute_transposition_features((3, 5, 7), (0, 1, 2))
        self.assertIsInstance(res, dict)
        self.assertEqual(res["edit"], 0)
        self.assertEqual(res["rot"], -1)
        res = compute_transposition_features((3, 5, 7), (2, 1, 0))
        self.assertEqual(res["edit"], 2)
        self.assertEqual(res["rot"], 0)
        self.assertEqual(res["rev"], 1)

    def test_cost(self):
        res = predict_transposition_cost((3, 5, 7), (0, 1, 2))
        self.assertIsInstance(res, float)
        self.assertGreater(res, 0)
        for shape in [(3, 5, 7), (30, 50, 70)]:
            for perm in permutations([0, 1, 2]):
                p = tuple(perm)
                cost = predict_transposition_cost(shape, p)
                if p[-1] == 2:
                    self.assertEqual(cost, 0)

    def test_edit_distance(self):
        r = _edit_distance("", "a")
        self.assertEqual(r, 1)
        r = _edit_distance("a", "")
        self.assertEqual(r, 1)
        r = _edit_distance("a", "ab")
        self.assertEqual(r, 1)


if __name__ == "__main__":
    unittest.main()
