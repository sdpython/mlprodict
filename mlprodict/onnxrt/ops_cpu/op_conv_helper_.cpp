// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef SKIP_PYTHON
//#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include <numpy/arrayobject.h>

#include "op_common_.hpp"

#if USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
#endif


template <typename T>
void im2col_1d_inplace(
        const py::array_t<T, py::array::c_style | py::array::forcecast>& result,
        const py::array_t<T, py::array::c_style | py::array::forcecast>& data,
        const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& kernel_shape,
        T fill_value) {
            
    std::vector<int64_t> data_shape;
    arrayshape2vector(data_shape, data);
    if (data_shape.size() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions: ", data_shape.size(), "."));
    if (kernel_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions: ", data_shape.size(), "."));
    const int64_t* p_kernel_shape = kernel_shape.data();
    
    std::vector<ssize_t> result_shape{data_shape[0], p_kernel_shape[0]};
    int64_t result_size = data_shape[0] * p_kernel_shape[0];

    T* p_result = (T*)result.data();

    // use AVX and parallelisation to be more efficient.
    const T* begin = data.data();
    size_t N = (size_t)data_shape[0];
    size_t k = p_kernel_shape[0];
    size_t lag = k / 2;
    ssize_t d;
    if (k >= N) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < k; ++j) {
                d = i + j - lag;
                p_result[i * k + j] = d < 0 ? fill_value : (
                    d >= (int)N ? fill_value : begin[d]);
            }
        }
    }
    else {
        size_t Nk = N - k;
        size_t i;
        for (i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                d = i + j - lag;
                p_result[i * k + j] = d < 0 ? fill_value : (
                    d >= (int)N ? fill_value : begin[d]);
            }
        }
        for(; i < Nk; ++i) {
            d = i - lag;
            std::copy(begin + d, begin + d + k, p_result + i * k);
        }
        for(; i < N; ++i) {
            for (size_t j = 0; j < k; ++j) {
                d = i + j - lag;
                p_result[i * k + j] = d < 0 ? fill_value : (
                    d >= (int)N ? fill_value : begin[d]);
            }
        }
    }
}


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_conv_helper_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Helpers for convolution functions."
    #else
    R"pbdoc(Helpers for convolution functions, inspired from
`conv_transpose.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv_transpose.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    m.def("im2col_1d_inplace_float", &im2col_1d_inplace<float>, "Applies im2col_1d on a single vector.",
          py::arg("result"), py::arg("data"), py::arg("kernel_shape"), py::arg("fill_value"),
          R"pbdoc(The function duplicates the one dimensional tensor so that 
the convolution can be done through a matrix multiplication. It returns 
a matrix `Nxk` where *N* is the tensor dimension and *k* the kernal shape.)pbdoc");
}

#endif
