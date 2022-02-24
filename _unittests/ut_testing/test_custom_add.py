"""
@brief      test log(time=8s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.experimental_c_impl.experimental_c import (  # pylint: disable=E0611
    BroadcastMatrixAddLeftInplaceDouble,
    BroadcastMatrixAddLeftInplaceFloat,
    BroadcastMatrixAddLeftInplaceInt64)


class TestCustomAdd(ExtTestCase):

    add_dtypes = {
        numpy.float64: BroadcastMatrixAddLeftInplaceDouble,
        numpy.float32: BroadcastMatrixAddLeftInplaceFloat,
        numpy.int64: BroadcastMatrixAddLeftInplaceInt64
    }

    def _common_broadcast_matrix(self, dt):
        with self.subTest(dtype=dt):
            fct = TestCustomAdd.add_dtypes[dt]

            m1 = numpy.array([1, 2, 3, 4, 5, 6], dtype=dt).reshape((-1, 2))
            m2 = numpy.array([1, 2], dtype=dt).reshape((1, 2))
            m3 = m1 + m2
            fct(m1, m2)
            self.assertEqualArray(m3, m1)

            m1 = numpy.array([1, 2, 3, 4, 5, 6], dtype=dt).reshape((-1, 3))
            m2 = numpy.array([1, 2], dtype=dt).reshape((2, 1))
            m3 = m1 + m2
            fct(m1, m2)
            self.assertEqualArray(m3, m1)

            m1 = numpy.array([1, 2, 3, 4, 5, 6], dtype=dt).reshape((-1, 3))
            m2 = numpy.array([1, 2], dtype=dt).reshape((2, ))
            m3 = m1 + m2.reshape((2, 1))
            fct(m1, m2)
            self.assertEqualArray(m3, m1)

    def test_broadcast_matrix(self):
        for dt in [numpy.float64, numpy.float32, numpy.int64]:
            self._common_broadcast_matrix(dt)


if __name__ == "__main__":
    # TestEinsum().test_np_test_broadcasting_dot_cases1()
    unittest.main()
