"""
@brief      test log(time=2s)
"""
import os
import unittest
import pickle
import numpy
from scipy.sparse import csr_matrix
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.tools.code_helper import (
    debug_print, debug_dump, numpy_min, numpy_max)


class TestCodeHelper(ExtTestCase):

    def test_debug_print(self):
        _, out, err = self.capture(
            lambda: debug_print('r', numpy.array([0, 1], dtype=numpy.float32), {}))
        self.assertIn("'r'", out)
        self.assertEmpty(err)

    def test_debug_dump(self):
        temp = get_temp_folder(__file__, "temp_debug_dump")
        obj = {'in': [numpy.array([0, 1]), numpy.array([1, 2])],
               'out': [numpy.array([0, numpy.nan])]}
        _, out, __ = self.capture(
            lambda: debug_dump("rrr", obj, temp))
        self.assertIn("NAN-notin-out", out)
        files = os.listdir(temp)
        self.assertEqual(len(files), 1)
        with open(os.path.join(temp, files[0]), 'rb') as f:
            obj2 = pickle.load(f)
        self.assertEqual(list(obj.keys()), list(obj2.keys()))

        obj = [numpy.array([0, 1]), None, numpy.array([1, 2, numpy.nan])]
        _, out, __ = self.capture(
            lambda: debug_dump("rrr", obj, temp))
        self.assertIn("NAN  2 rrr (3,)", out)

    def test_min_max(self):
        a = numpy_max(numpy.array([1, 3]))
        b = numpy_min(numpy.array([1, 3]))
        self.assertEqual(a, 3)
        self.assertEqual(b, 1)
        a = numpy_max(csr_matrix(numpy.array([1, 3])))
        b = numpy_min(csr_matrix(numpy.array([1, 3])))
        self.assertEqual(a, 3)
        self.assertEqual(b, 1)
        a = numpy_max(numpy.array(['a1', 'b3']))
        b = numpy_min(numpy.array(['a1', 'b3']))
        self.assertEqual(a, "'b3'")
        self.assertEqual(b, "'a1'")
        a = numpy_max(numpy.array(['aaaaaaaaaaaa1', 'b3']))
        b = numpy_min(numpy.array(['aaaaaaaaaaaa1', 'b3']))
        self.assertEqual(a, "'b3'")
        self.assertEqual(b, "'aaaaaaaaaa...'")
        for dtype in [numpy.int8, numpy.int16, numpy.float32,
                      numpy.float64, numpy.float16, numpy.int32,
                      numpy.int64, numpy.uint8, numpy.uint16,
                      numpy.uint32, numpy.uint64]:
            a = numpy_max(numpy.array([1, 3], dtype=dtype))
            b = numpy_min(numpy.array([1, 3], dtype=dtype))
            self.assertEqual(a, 3)
            self.assertEqual(b, 1)
        a = numpy_max([1, 3])
        b = numpy_min([1, 3])
        self.assertEqual(a, '?')
        self.assertEqual(b, '?')


if __name__ == "__main__":
    unittest.main()
