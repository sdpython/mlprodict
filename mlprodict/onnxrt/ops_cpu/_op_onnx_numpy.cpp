// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <thread>
#include <iterator>

#ifndef SKIP_PYTHON
//#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include <numpy/arrayobject.h>

#if USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
#endif

#include "op_common_.hpp"


template<typename NTYPE>
py::array_t<NTYPE> array_feature_extractor(py::array_t<NTYPE> data,
                                           py::array_t<int64_t> indices_) {
                                               
    std::vector<int64_t> x_shape, y_shape;

    arrayshape2vector(x_shape, data);
    ssize_t x_num_dims = x_shape.size();
                                               
    const NTYPE* x_data = data.data();
    const int64_t* indices = indices_.data();

    if (x_num_dims == 0)
        throw std::runtime_error("data cannot be empty.");

    arrayshape2vector(y_shape, indices_);
    ssize_t stride = x_shape[x_num_dims - 1];
    ssize_t num_indices = flattened_dimension(y_shape);

    if (num_indices == 0)
        throw std::runtime_error("indices cannot be empty.");

    for (ssize_t i = 0; i < num_indices; ++i)
        if (indices[i] >= (int64_t)stride)
            throw std::runtime_error(
                "Invalid Y argument: index is out of range");

    std::vector<ssize_t> z_shape;
    if (x_num_dims == 1) {
        z_shape.push_back(1);
        z_shape.push_back(num_indices);
    }
    else {
        z_shape = x_shape;
        z_shape[x_num_dims - 1] = num_indices;
    }

    std::vector<NTYPE> z_vector(flattened_dimension(z_shape));
    NTYPE* z_data = z_vector.data();

    int64_t x_size_until_last_dim = flattened_dimension(x_shape, x_num_dims - 1);
    const int64_t * indices_end = indices + num_indices;
    const int64_t * iti;
    for (ssize_t i = 0; i < x_size_until_last_dim; ++i) {
        for (iti = indices; iti != indices_end; ++iti) {
            *z_data++ = x_data[*iti];
        }
        x_data += stride;
    }
    std::vector<ssize_t> strides;
    shape2strides(z_shape, strides, (NTYPE)0);
    
    return py::array_t<NTYPE>(
        py::buffer_info(
            &z_vector[0],
            sizeof(NTYPE),
            py::format_descriptor<NTYPE>::format(),
            z_shape.size(),
            z_shape,        /* shape of the matrix       */
            strides         /* strides for each axis     */
        ));    
}


py::array_t<float> array_feature_extractor_float(
    py::array_t<float> data, py::array_t<int64_t> indices) {
    return array_feature_extractor(data, indices);
}        


py::array_t<double> array_feature_extractor_double(
    py::array_t<double> data, py::array_t<int64_t> indices) {
    return array_feature_extractor(data, indices);
}        


py::array_t<int64_t> array_feature_extractor_int64(
    py::array_t<int64_t> data, py::array_t<int64_t> indices) {
    return array_feature_extractor(data, indices);
}        


#ifndef SKIP_PYTHON

PYBIND11_MODULE(_op_onnx_numpy, m) {
	m.doc() =
    #if defined(__APPLE__)
    "C++ helpers of ONNX operators."
    #else
    R"pbdoc(C++ helpers of ONNX operators.)pbdoc"
    #endif
    ;

    m.def("array_feature_extractor_float", &array_feature_extractor_float,
            "C++ implementation of operator ArrayFeatureExtractor for float32.");
    m.def("array_feature_extractor_double", &array_feature_extractor_double,
            "C++ implementation of operator ArrayFeatureExtractor for float64.");
    m.def("array_feature_extractor_int64", &array_feature_extractor_int64,
            "C++ implementation of operator ArrayFeatureExtractor for int64.");
}

#endif
