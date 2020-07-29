"""
@brief      test log(time=8s)
"""
import os
import unittest
import pickle
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.tools.code_helper import debug_print, debug_dump


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


if __name__ == "__main__":
    unittest.main()
