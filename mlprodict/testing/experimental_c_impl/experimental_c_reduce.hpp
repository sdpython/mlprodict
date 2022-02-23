#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#include "experimental_c_reduce.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>


template <typename NTYPE>
void vector_add_pointer(NTYPE* acc, const NTYPE* x, size_t size) {
    for (; size != 0; ++acc, ++x, --size)
        *acc += *x;
}

template <>
void vector_add_pointer(float* acc, const float* x, size_t size) {
    // _mm_store_ps fails if acc not aligned.
    // _mm_storeu_ps does not need alignment.
#if defined(__AVX__)
    if (size > 8) {
        for (; size > 8; acc += 8, x += 8, size -= 8) {
            _mm256_storeu_ps(acc, _mm256_add_ps(_mm256_loadu_ps(acc), _mm256_loadu_ps(x)));
        }
    }
#else
    if (size > 4) {
        for (; size > 4; acc += 4, x += 4, size -= 4) {
            _mm_storeu_ps(acc, _mm_add_ps(_mm_loadu_ps(acc), _mm_loadu_ps(x)));
        }
    }
#endif
    for (; size != 0; ++acc, ++x, --size)
        *acc += *x;
}

#ifndef SKIP_PYTHON

// This function assumes x is a 2D matrix to be reduced on the first axis.
template<typename NTYPE>
py::array_t<NTYPE> custom_reducesum_rk(py::array_t<NTYPE, py::array::c_style | py::array::forcecast> x,
    int nthread) {
    std::vector<int64_t> x_shape;
    arrayshape2vector(x_shape, x);
    if (x_shape.size() != 2)
        throw std::runtime_error("Input array must have two dimensions.");
    if (flattened_dimension(x_shape) == 0)
        throw std::runtime_error("Input array must not be empty.");

    int64_t N = x_shape[1];
    std::vector<NTYPE> y_vector(N);
    // int64_t Nred = x_shape[0];
    const NTYPE* x_data = x.data();
    // const NTYPE* x_data_end = x_data + x_shape[0] * x_shape[1]; 
    NTYPE* y_data = y_vector.data();

#if USE_OPENMP
    if (nthread == 1 || N <= nthread * 2) {
#endif
        int64_t n_rows = x_shape[0];
        // NTYPE *y_data_end = y_data + N;
        memcpy(y_data, x_data, N * sizeof(NTYPE));
        for (int64_t row = 1; row < n_rows; ++row) {
            vector_add_pointer(y_data, x_data + row * N, N);
        }
#if USE_OPENMP
    }
    else {
        if (nthread > 1)
            omp_set_num_threads(nthread);
        else
            nthread = omp_get_num_procs();

        int64_t batch_size = N / nthread / 2;
        int64_t n_rows = x_shape[0];
        batch_size = batch_size < 4 ? 4 : batch_size;
        batch_size = batch_size > 1024 ? 1024 : batch_size;
        int64_t batch = N / batch_size + (N % batch_size > 0 ? 1 : 0);
        memcpy(y_data, x_data, N * sizeof(NTYPE));

#pragma omp parallel for
        for (int64_t b = 0; b < batch; ++b) {
            int64_t begin = batch_size * b;
            int64_t end = begin + batch_size < N ? begin + batch_size : N;
            for (int64_t row = 1; row < n_rows; ++row) {
                vector_add_pointer(y_data + begin, x_data + row * N + begin, end - begin);
            }
        }
    }
#endif

    std::vector<int64_t> y_shape{ N };
    std::vector<ssize_t> strides;
    shape2strides(y_shape, strides, (NTYPE)0);

    return py::array_t<NTYPE>(
        py::buffer_info(
            &y_vector[0],
            sizeof(NTYPE),
            py::format_descriptor<NTYPE>::format(),
            y_shape.size(),
            y_shape,        /* shape of the matrix       */
            strides         /* strides for each axis     */
        ));
}

py::array_t<float> custom_reducesum_rk_float(py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int nthread) {
    return custom_reducesum_rk(x, nthread);
}

py::array_t<double> custom_reducesum_rk_double(py::array_t<double, py::array::c_style | py::array::forcecast> x,
    int nthread) {
    return custom_reducesum_rk(x, nthread);
}

#endif

void experimental_ut_reduce();
