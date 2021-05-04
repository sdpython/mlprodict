"""
@file
@brief Direct calls to libraries :epkg:`BLAS` and :epkg:`LAPACK`.
"""
from libc.stdio cimport printf

import numpy
cimport numpy
cimport cython
numpy.import_array()
# cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas



cdef void dgemm_dot(numpy.ndarray[double, ndim=2, mode='c'] A,
                    numpy.ndarray[double, ndim=2, mode='c'] B,
                    int transA, int transB,
                    numpy.ndarray[double, ndim=2, mode='c'] C):
    """
    Calls gemm for a dot product. Avoids translation if possible.
    Does `A @ B`.
    """

    cdef:
        char ca = "T" if transA else "N"
        char cb = "T" if transB else "N"
        int lda = K if transA else M
        int ldb = K if transB else N
        int ldc = 0
        const double* pa = &A[0, 0]
        const double* pb = &B[0, 0]
        double* pc = &C[0, 0]
        int M = A.shape[1] if transA else A.shape[0]
        int N = B.shape[0] if transB else B.shape[0]
        int K = A.shape[0] if transA else A.shape[1]
        double one = 1.
        double zero = 0.

    cython_blas.dgemm(&ca, &cb, &M, &N, &K, &one, pa, &lda, pb, &ldb, &zero, pb, &ldc)


cdef void sgemm_dot(numpy.ndarray[float, ndim=2, mode='c'] A,
                    numpy.ndarray[float, ndim=2, mode='c'] B,
                    int transA, int transB,
                    numpy.ndarray[float, ndim=2, mode='c'] C):
    """
    Calls gemm for a dot product. Avoids translation if possible.
    Does `A @ B`.
    """

    cdef:
        char ca = "T" if transA else "N"
        char cb = "T" if transB else "N"
        int lda = K if transA else M
        int ldb = K if transB else N
        int ldc = 0
        const float* pa = &A[0, 0]
        const float* pb = &B[0, 0]
        float* pc = &C[0, 0]
        int M = A.shape[1] if transA else A.shape[0]
        int N = B.shape[0] if transB else B.shape[0]
        int K = A.shape[0] if transA else A.shape[1]
        float one = 1.
        float zero = 0.

    cython_blas.sgemm(&ca, &cb, &M, &N, &K, &one, pa, &lda, pb, &ldb, &zero, pb, &ldc)
