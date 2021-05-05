"""
@file
@brief Direct calls to libraries :epkg:`BLAS` and :epkg:`LAPACK`.
"""
import numpy
from scipy.linalg.blas import sgemm, dgemm  # pylint: disable=E0611
from .direct_blas_lapack import (  # pylint: disable=E0401,E0611
    dgemm_dot, sgemm_dot)


def pygemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Pure python implementatin of GEMM.
    """
    if len(A.shape) != 1:
        raise ValueError("A must be a vector.")
    if len(B.shape) != 1:
        raise ValueError("B must be a vector.")
    if len(C.shape) != 1:
        raise ValueError("C must be a vector.")
    if A.shape[0] != M * K:
        raise ValueError(
            "Dimension mismatch for A.shape=%r M=%r N=%r K=%r." % (
                A.shape, M, N, K))
    if B.shape[0] != N * K:
        raise ValueError(
            "Dimension mismatch for B.shape=%r M=%r N=%r K=%r." % (
                B.shape, M, N, K))
    if C.shape[0] != N * M:
        raise ValueError(
            "Dimension mismatch for C.shape=%r M=%r N=%r K=%r." % (
                C.shape, M, N, K))

    if transA:
        a_i_stride = lda
        a_k_stride = 1
    else:
        a_i_stride = 1
        a_k_stride = lda

    if transB:
        b_j_stride = 1
        b_k_stride = ldb
    else:
        b_j_stride = ldb
        b_k_stride = 1

    c_i_stride = 1
    c_j_stride = ldc

    n_loop = 0
    for j in range(N):
        for i in range(M):
            total = 0
            for k in range(K):
                n_loop += 1
                a_index = i * a_i_stride + k * a_k_stride
                if a_index >= A.shape[0]:
                    raise IndexError(
                        "A: i=%d a_index=%d >= %d "
                        "(a_i_stride=%d a_k_stride=%d)" % (
                            i, a_index, A.shape[0], a_i_stride, a_k_stride))
                a_val = A[a_index]

                b_index = j * b_j_stride + k * b_k_stride
                if b_index >= B.shape[0]:
                    raise IndexError(
                        "B: j=%d b_index=%d >= %d "
                        "(a_i_stride=%d a_k_stride=%d)" % (
                            j, b_index, B.shape[0], b_j_stride, b_k_stride))
                b_val = B[b_index]

                mult = a_val * b_val
                total += mult

            c_index = i * c_i_stride + j * c_j_stride
            if c_index >= C.shape[0]:
                raise IndexError("C: %d >= %d" % (c_index, C.shape[0]))
            C[c_index] = alpha * total + beta * C[c_index]

    if n_loop != M * N * K:
        raise RuntimeError(
            "Unexpected number of loops: %d != %d = (%d * %d * %d) "
            "lda=%d ldb=%d ldc=%d" % (
                n_loop, M * N * K, M, N, K, lda, ldb, ldc))


def gemm_dot(A, B, transA=False, transB=False):
    """
    Implements dot product with gemm when possible.

    :param A: first matrix
    :param B: second matrix
    :param transA: is first matrix transposed?
    :param transB: is second matrix transposed?
    """
    if A.dtype != B.dtype:
        raise TypeError(
            "Matrices A and B must have the same dtype not "
            "%r, %r." % (A.dtype, B.dtype))
    if len(A.shape) != 2:
        raise ValueError(
            "Matrix A does not have 2 dimensions but %d." % len(A.shape))
    if len(B.shape) != 2:
        raise ValueError(
            "Matrix B does not have 2 dimensions but %d." % len(B.shape))

    def _make_contiguous_(A, B):
        if not A.flags['C_CONTIGUOUS']:
            A = numpy.ascontiguousarray(A)
        if not B.flags['C_CONTIGUOUS']:
            B = numpy.ascontiguousarray(B)
        return A, B

    all_dims = A.shape + B.shape
    square = min(all_dims) == max(all_dims)

    if transA:
        if transB:
            if A.dtype == numpy.float32:
                if square:
                    C = numpy.zeros((A.shape[1], B.shape[0]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    sgemm_dot(B, A, True, True, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[1], B.shape[0]), dtype=A.dtype)
                    return sgemm(1, A, B, 0, C, 1, 1, 1)
            if A.dtype == numpy.float64:
                if square:
                    C = numpy.zeros((A.shape[1], B.shape[0]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    dgemm_dot(B, A, True, True, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[1], B.shape[0]), dtype=A.dtype)
                    return dgemm(1, A, B, 0, C, 1, 1, 1)
            return A.T @ B.T
        else:
            if A.dtype == numpy.float32:
                if square:
                    C = numpy.zeros((A.shape[1], B.shape[1]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    sgemm_dot(B, A, False, True, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[1], B.shape[1]), dtype=A.dtype)
                    return sgemm(1, A, B, 0, C, 1, 0, 1)
            if A.dtype == numpy.float64:
                if square:
                    C = numpy.zeros((A.shape[1], B.shape[1]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    dgemm_dot(B, A, False, True, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[1], B.shape[1]), dtype=A.dtype)
                    return dgemm(1, A, B, 0, C, 1, 0, 1)
            return A.T @ B
    else:
        if transB:
            if A.dtype == numpy.float32:
                if square:
                    C = numpy.zeros((A.shape[0], B.shape[0]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    sgemm_dot(B, A, True, False, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[0], B.shape[0]), dtype=A.dtype)
                    return sgemm(1, A, B, 0, C, 0, 1, 1)
            if A.dtype == numpy.float64:
                if square:
                    C = numpy.zeros((A.shape[0], B.shape[0]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    dgemm_dot(B, A, True, False, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[0], B.shape[0]), dtype=A.dtype)
                    return dgemm(1, A, B, 0, C, 0, 1, 1)
            return A @ B.T
        else:
            if A.dtype == numpy.float32:
                if square:
                    C = numpy.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    sgemm_dot(B, A, False, False, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
                    return sgemm(1, A, B, 0, C, 0, 0)
            if A.dtype == numpy.float64:
                if square:
                    C = numpy.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
                    A, B = _make_contiguous_(A, B)
                    dgemm_dot(B, A, False, False, C)
                    return C
                else:
                    C = numpy.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
                    return dgemm(1, A, B, 0, C, 0, 0, 1)
            return A @ B
