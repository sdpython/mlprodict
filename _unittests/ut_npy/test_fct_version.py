# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.onnx_version import FctVersion


class TestOnnxVersion(ExtTestCase):

    def test_version(self):
        version = FctVersion(None, None)
        r = repr(version)
        self.assertEqual(r, "FctVersion(None, None)")
        self.assertEqual(version.args, None)
        self.assertEqual(version.kwargs, None)
        self.assertEqual(len(version), 0)
        self.assertEqual(version.as_tuple(), tuple())
        self.assertEqual(version.as_string(), '_')

    def test_version_string(self):
        version = FctVersion((numpy.float32, ), ('constant', ))
        self.assertEqual(version.as_string(), 'float32___constant')
        version = FctVersion((numpy.float32, ), None)
        self.assertEqual(version.as_string(), 'float32__')
        version = FctVersion((numpy.float32, ), None)
        self.assertEqual(repr(version), "FctVersion((numpy.float32,), None)")

    def test_version_exc(self):
        self.assertRaise(lambda: FctVersion([], None)._check_(),  # pylint: disable=W0212
                         TypeError)
        self.assertRaise(lambda: FctVersion(None, [])._check_(),  # pylint: disable=W0212
                         TypeError)
        version = FctVersion(None, None)

        def do(v):
            v.args = []
        self.assertRaise(lambda: do(version), AttributeError)

    def test_version_dictionary(self):
        def fct():
            pass
        keys = [
            FctVersion(None, None),
            FctVersion((numpy.float32, ), ('constant', )),
            FctVersion((numpy.float32, ), (fct, ))
        ]
        dc = {}
        for k in keys:
            dc[k] = None
        self.assertEqual(len(dc), len(keys))


if __name__ == "__main__":
    unittest.main()
