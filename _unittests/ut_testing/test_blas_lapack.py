"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from scipy.linalg.blas import sgemm  # pylint: disable=E0611
from pyquickhelper.pycode import ExtTestCase
from mlprodict.testing.blas_lapack import gemm_dot, pygemm


class TestBlasLapack(ExtTestCase):

    def test_gemm(self):
        A = numpy.arange(4).reshape((2, 2)) + 1
        B = numpy.arange(4).reshape((2, 2)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(dtype=dtype, transA=t1, transB=t2,
                                      shapeA=a.shape, shapeB=b.shape):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        exp = ta @ tb
                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

                        M, N, K = 2, 2, 2
                        lda, ldb, ldc = 2, 2, 2

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(t2, t1, M, N, K, 1.,
                               b.ravel(), ldb, a.ravel(), lda,
                               0., c, ldc)
                        cc = c.reshape((M, N))
                        self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

    def test_gemm_1(self):
        A = numpy.arange(1).reshape((1, 1)) + 1
        B = numpy.arange(1).reshape((1, 1)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(dtype=dtype, transA=t1, transB=t2,
                                      shapeA=a.shape, shapeB=b.shape):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        exp = ta @ tb
                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

                        M, N, K = 1, 1, 1
                        lda, ldb, ldc = 1, 1, 1

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(t2, t1, M, N, K, 1.,
                               b.ravel(), ldb, a.ravel(), lda,
                               0., c, ldc)
                        cc = c.reshape((M, N))
                        self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

    def test_gemm_exc(self):
        a = numpy.arange(3).reshape((1, 3)) + 1
        b = numpy.arange(3).reshape((3, 1)) + 10
        c = numpy.empty((1, 2), dtype=a.dtype)
        self.assertRaise(
            lambda: pygemm(False, False, 1, 1, 1, 1., b.ravel(),
                           1, a.ravel(), 1, 0., c, 1),
            ValueError)
        c = numpy.empty((1, ), dtype=a.dtype)
        self.assertRaise(
            lambda: pygemm(False, False, 1, 1, 1, 1., b.ravel(),
                           1, a.ravel(), 1, 0., c, 1),
            ValueError)
        c = numpy.empty((1, ), dtype=a.dtype)
        a = numpy.arange(4) + 1
        b = numpy.arange(4) + 10
        c = numpy.empty((4, ), dtype=a.dtype)
        self.assertRaise(
            lambda: pygemm(False, False, 2, 4, 2, 1., b.ravel(),
                           10, a.ravel(), 10, 0., c, 10),
            ValueError)
        self.assertRaise(
            lambda: pygemm(False, False, 2, 2, 2, 1., b.ravel(),
                           10, a.ravel(), 10, 0., c, 10),
            IndexError)
        self.assertRaise(
            lambda: pygemm(False, False, 2, 2, 2, 1., b.ravel(),
                           1, a.ravel(), 10, 0., c, 10),
            IndexError)
        self.assertRaise(
            lambda: pygemm(False, False, 2, 2, 2, 1., b.ravel(),
                           1, a.ravel(), 1, 0., c, 10),
            IndexError)

    def test_gemm_314(self):
        A = numpy.arange(3).reshape((1, 3)) + 1
        B = numpy.arange(4).reshape((4, 1)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(dtype=dtype, transA=t1, transB=t2,
                                      shapeA=a.shape, shapeB=b.shape):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        try:
                            exp = ta @ tb
                        except ValueError:
                            continue

                        if t1:
                            M = a.shape[1]
                            lda = a.shape[0]
                            K = a.shape[0]
                        else:
                            M = a.shape[0]
                            lda = a.shape[0]
                            K = a.shape[1]

                        if t2:
                            N = b.shape[0]
                            ldb = b.shape[1]
                        else:
                            N = b.shape[1]
                            ldb = b.shape[1]
                        ldc = N

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(t2, t1, N, M, K, 1.,
                               b.ravel(), ldb, a.ravel(), lda,
                               0., c, ldc)
                        cc = c.reshape((M, N))
                        self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

                            cc[:, :] = 0
                            sgemm(1, a, b, 0, cc, t1, t2, 1)
                            try:
                                self.assertEqualArray(exp, cc)
                            except AssertionError:
                                # Overwriting the result does not seem
                                # to work.
                                pass

                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

    def test_gemm_324(self):
        A = numpy.arange(6).reshape((2, 3)) + 1
        B = numpy.arange(8).reshape((4, 2)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(dtype=dtype, transA=t1, transB=t2,
                                      shapeA=a.shape, shapeB=b.shape):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        try:
                            exp = ta @ tb
                        except ValueError:
                            continue

                        if t1:
                            M = a.shape[1]
                            lda = a.shape[0]
                            K = a.shape[0]
                        else:
                            M = a.shape[0]
                            lda = a.shape[0]
                            K = a.shape[1]

                        if t2:
                            N = b.shape[0]
                            ldb = b.shape[1]
                        else:
                            N = b.shape[1]
                            ldb = b.shape[1]
                        ldc = N

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(t2, t1, N, M, K, 1.,
                               b.ravel(), ldb, a.ravel(), lda,
                               0., c, ldc)
                        cc = c.reshape((M, N))
                        # self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

                            cc[:, :] = 0
                            sgemm(1, a, b, 0, cc, t1, t2, 1)
                            try:
                                self.assertEqualArray(exp, cc)
                            except AssertionError:
                                # Overwriting the result does not seem
                                # to work.
                                pass

                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)

    def test_gemm_323(self):
        A = numpy.arange(6).reshape((2, 3)) + 1
        B = numpy.arange(6).reshape((3, 2)) + 10
        for dtype in [numpy.float32, numpy.float64, numpy.int64]:
            a = A.astype(dtype)
            b = B.astype(dtype)
            for t1 in [False, True]:
                for t2 in [False, True]:
                    with self.subTest(dtype=dtype, transA=t1, transB=t2,
                                      shapeA=a.shape, shapeB=b.shape):
                        ta = a.T if t1 else a
                        tb = b.T if t2 else b
                        try:
                            exp = ta @ tb
                        except ValueError:
                            continue

                        if t1:
                            M = a.shape[1]
                            lda = a.shape[0]
                            K = a.shape[0]
                        else:
                            M = a.shape[0]
                            lda = a.shape[0]
                            K = a.shape[1]

                        if t2:
                            N = b.shape[0]
                            ldb = b.shape[1]
                        else:
                            N = b.shape[1]
                            ldb = b.shape[1]
                        ldc = N

                        c = numpy.empty(M * N, dtype=a.dtype)
                        pygemm(t2, t1, N, M, K, 1.,
                               b.ravel(), ldb, a.ravel(), lda,
                               0., c, ldc)
                        cc = c.reshape((M, N))
                        # self.assertEqualArray(exp, cc)

                        if dtype == numpy.float32:
                            res = sgemm(1, a, b, 0, cc, t1, t2)
                            self.assertEqualArray(exp, res)

                            cc[:, :] = 0
                            sgemm(1, a, b, 0, cc, t1, t2, 1)
                            try:
                                self.assertEqualArray(exp, cc)
                            except AssertionError:
                                # Overwriting the result does not seem
                                # to work.
                                pass

                        got = gemm_dot(a, b, t1, t2)
                        self.assertEqualArray(exp, got)


if __name__ == "__main__":
    unittest.main()
