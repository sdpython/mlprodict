"""
@brief      test log(time=2s)
"""
import unittest
import numpy
import pandas
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.ops_cpu.op_zipmap import ZipMapDictionary, ArrayZipMapDictionary


class TestOnnxrtSimpleZipMap(ExtTestCase):

    def test_onnxt_zipmap(self):
        keys = ['a', 'b']
        values = ['a1', 'b1']
        zm = ZipMapDictionary(ZipMapDictionary.build_rev_keys(keys), values)
        df = pandas.DataFrame([zm])
        self.assertEqual(len(zm), 2)
        items = list(zm.items())
        self.assertEqual(items, [('a', 'a1'), ('b', 'b1')])
        for k in keys:
            self.assertIn(k, zm)
        self.assertEqual(list(df.columns), ['a', 'b'])
        self.assertEqual(list(sorted(zm.keys())), ['a', 'b'])
        self.assertEqual(list(sorted(zm.values())), ['a1', 'b1'])
        self.assertIn('a', zm)
        self.assertEqual(zm['a'], 'a1')
        self.assertEqual(zm['b'], 'b1')
        self.assertEqual(df.iloc[0, 0], 'a1')
        self.assertEqual(df.iloc[0, 1], 'b1')

    def test_onnxt_zipmap_mat(self):
        keys = ['a', 'b']
        values = numpy.array([['a1', 'b1']])
        zm = ZipMapDictionary(ZipMapDictionary.build_rev_keys(keys), 0, values)
        df = pandas.DataFrame([zm])
        self.assertEqual(len(zm), 2)
        items = list(zm.items())
        self.assertEqual(items, [('a', 'a1'), ('b', 'b1')])
        for k in keys:
            self.assertIn(k, zm)
        self.assertEqual(list(df.columns), ['a', 'b'])
        self.assertEqual(list(sorted(zm.keys())), ['a', 'b'])
        self.assertEqual(list(sorted(zm.values())), ['a1', 'b1'])
        self.assertIn('a', zm)
        self.assertEqual(zm['a'], 'a1')
        self.assertEqual(zm['b'], 'b1')
        self.assertEqual(df.iloc[0, 0], 'a1')
        self.assertEqual(df.iloc[0, 1], 'b1')

    def test_onnxt_zipmap_mat_exc(self):
        keys = ['a', 'b']
        values = numpy.array([['a1', 'b1']])
        zm = ZipMapDictionary(ZipMapDictionary.build_rev_keys(keys), 0, values)

        def _set():
            zm[6] = 'r'

        self.assertRaise(lambda: _set(), RuntimeError)

    def test_ufunc(self):
        keys = ['a', 'b']
        rk = ZipMapDictionary.build_rev_keys(keys)
        mat = numpy.array([[0, 1], [2, 3]])
        uf = numpy.frompyfunc(
            lambda _, d=rk, m=mat: ZipMapDictionary(d, _, m), 1, 1)
        res = uf(numpy.arange(mat.shape[0]))
        self.assertEqual(res[0]['a'], 0)
        self.assertEqual(res[0]['b'], 1)
        self.assertEqual(res[1]['a'], 2)
        self.assertEqual(res[1]['b'], 3)
        df = pandas.DataFrame(list(res))
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.iloc[0, 1], 1)

    def test_onnxt_arrayzipmap_mat(self):
        keys = ['a', 'b']
        values = numpy.array([['a1', 'b1']])
        azm = ArrayZipMapDictionary(
            ZipMapDictionary.build_rev_keys(keys), values)
        df = pandas.DataFrame(azm)
        self.assertEqual(df.shape, (1, 2))
        items = list(azm)
        self.assertIsInstance(azm[0], ZipMapDictionary)
        self.assertIsInstance(items[0], ZipMapDictionary)
        d = items[0].asdict()
        self.assertEqual(d, {'a': 'a1', 'b': 'b1'})
        self.assertEqual(list(items[0].items()), [('a', 'a1'), ('b', 'b1')])
        self.assertEqual(len(azm), 1)

        def _set():
            azm[0] = 5

        self.assertRaise(lambda: _set(), RuntimeError)
        self.assertIsInstance(azm.values, numpy.ndarray)
        self.assertEqual(azm.columns, ['a', 'b'])


if __name__ == "__main__":
    unittest.main()
