"""
@brief      test log(time=8s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.einsum_bench import einsum_benchmark


class TestEinsumBenchmark(ExtTestCase):

    def test_benchmark1(self):
        for rt in ['numpy', 'python', 'onnxruntime']:
            with self.subTest(rt=rt):
                res = list(einsum_benchmark(shape=5))
                self.assertEqual(len(res), 2)

    def test_benchmark2(self):
        for rt in ['numpy', 'python', 'onnxruntime']:
            with self.subTest(rt=rt):
                res = list(einsum_benchmark(shape=[5, 6]))
                self.assertEqual(len(res), 4)

    def test_benchmark1_shape(self):
        for rt in ['numpy', 'python', 'onnxruntime']:
            with self.subTest(rt=rt):
                res = list(einsum_benchmark(shape=[(5, 5, 5), (5, 5)]))
                self.assertEqual(len(res), 2)

    def test_benchmarkn(self):
        for rt in ['numpy']:
            with self.subTest(rt=rt):
                res = list(einsum_benchmark(shape=5, perm=True))
                self.assertEqual(len(res), 48)


if __name__ == "__main__":
    unittest.main()
