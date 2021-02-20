// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <thread>
#include <iterator>
#include <queue>
#include <iostream>
#include <algorithm>

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


/////////////////////////////////////////////
// begin: array_feature_extractor
/////////////////////////////////////////////


template<typename NTYPE>
py::array_t<NTYPE> array_feature_extractor(py::array_t<NTYPE, py::array::c_style | py::array::forcecast> data,
                                           py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices_) {
                                               
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
        z_shape.resize(x_shape.size());
        for(size_t i = 0; i < x_shape.size(); ++i)
            z_shape[i] = (ssize_t)x_shape[i];
        z_shape[x_num_dims - 1] = num_indices;
    }

    std::vector<NTYPE> z_vector(flattened_dimension(z_shape));
    NTYPE* z_data = z_vector.data();

    int64_t x_size_until_last_dim = flattened_dimension(x_shape, x_num_dims - 1);
    const int64_t * indices_end = indices + num_indices;
    const int64_t * iti;
    for (ssize_t i = 0; i < x_size_until_last_dim; ++i) {
        for (iti = indices; iti != indices_end; ++iti)
            *z_data++ = x_data[*iti];
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
        py::array_t<float, py::array::c_style | py::array::forcecast> data,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
    return array_feature_extractor(data, indices);
}        


py::array_t<double> array_feature_extractor_double(
        py::array_t<double, py::array::c_style | py::array::forcecast> data, 
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
    return array_feature_extractor(data, indices);
}        


py::array_t<int64_t> array_feature_extractor_int64(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> data,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
    return array_feature_extractor(data, indices);
}

/////////////////////////////////////////////
// end: array_feature_extractor
/////////////////////////////////////////////


/////////////////////////////////////////////
// begin: topk
/////////////////////////////////////////////


template <typename NTYPE>
struct HeapMax {
    using DataType = NTYPE;
    bool cmp1(const NTYPE& v1, const NTYPE& v2) const { return v1 > v2; }
    bool cmp(int64_t i1, int64_t i2, const NTYPE* ens, const int64_t* pos) const {
        return (ens[pos[i1]] < ens[pos[i2]]) ||
               ((pos[i1] > pos[i2]) && (ens[pos[i1]] == ens[pos[i2]]));
    }
};


template <typename NTYPE>
struct HeapMin {
    using DataType = NTYPE;
    bool cmp1(const NTYPE& v1, const NTYPE& v2) const { return v1 < v2; }
    bool cmp(int64_t i1, int64_t i2, const NTYPE* ens, const int64_t* pos) const {
        return (ens[pos[i1]] > ens[pos[i2]]) ||
               ((pos[i1] > pos[i2]) && (ens[pos[i1]] == ens[pos[i2]]));
    }
};


template <class HeapCmp>
void _heapify_up_position(const typename HeapCmp::DataType* ens, int64_t* pos,
                          size_t i, size_t k, const HeapCmp& heap_cmp) {
    size_t left, right;
    int64_t ch;
    while (true) {
        left = 2 * i + 1;
        right = left + 1;
        if (right < k) {
            if (heap_cmp.cmp(left, i, ens, pos) && !heap_cmp.cmp1(ens[pos[left]], ens[pos[right]])) {
                ch = pos[i];
                pos[i] = pos[left];
                pos[left] = ch;
                i = left;
            }
            else if (heap_cmp.cmp(right, i, ens, pos)) {
                ch = pos[i];
                pos[i] = pos[right];
                pos[right] = ch;
                i = right;
            }
            else
                break;
        }
        else if ((left < k) && heap_cmp.cmp(left, i, ens, pos)) {
            ch = pos[i];
            pos[i] = pos[left];
            pos[left] = ch;
            i = left;
        }
        else
            break;
    }
}


template <class HeapCmp>
void _topk_element(const typename HeapCmp::DataType* values, size_t k, size_t n,
                   int64_t* indices, bool sorted, const HeapCmp& heap_cmp) {
    if (n <= k && !sorted) {
        for (size_t i = 0; i < n; ++i, ++indices)
        *indices = i;
    } 
    else if (k == 1) {
        auto begin = values;
        auto end = values + n;
        *indices = 0;
        for (++values; values != end; ++values)
            *indices = heap_cmp.cmp1(*values, begin[*indices]) ? ((int64_t)(values - begin)) : *indices;
    } 
    else {
        indices[k - 1] = 0;

        size_t i = 0;
        for (; i < k; ++i) {
            indices[k - i - 1] = i;
            _heapify_up_position(values, indices, k - i - 1, k, heap_cmp);
        }
        for (; i < n; ++i) {
            if (heap_cmp.cmp1(values[i], values[indices[0]])) {
                indices[0] = i;
                _heapify_up_position(values, indices, 0, k, heap_cmp);
            }
        }
        if (sorted) {
            int64_t ech;
            i = k - 1;
            ech = indices[0];
            indices[0] = indices[i];
            indices[i] = ech;
            --i;
            for (; i > 0; --i) {
                _heapify_up_position(values, indices, 0, i + 1, heap_cmp);
                ech = indices[0];
                indices[0] = indices[i];
                indices[i] = ech;
            }
        }
    }
}


template <class HeapCmp>
void _topk_element_ptr(int64_t* pos, size_t k, const typename HeapCmp::DataType* values,
                       const std::vector<int64_t>& shape, bool sorted, ssize_t th_parallel) {
    HeapCmp heap_cmp;
    if (shape.size() == 1) {
        _topk_element(values, k, shape[0], pos, sorted, heap_cmp);
    }
    else {
        auto vdim = shape[shape.size() - 1];
        auto ptr = pos;

        if (shape[0] <= th_parallel) {
            auto tdim = flattened_dimension(shape);
            const typename HeapCmp::DataType* data = values;
            const typename HeapCmp::DataType* end = data + tdim;
            for (; data != end; data += vdim, ptr += k)
                _topk_element(data, k, vdim, ptr, sorted, heap_cmp);
        } 
        else {
            // parallelisation
            const typename HeapCmp::DataType* data = values;
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t nr = 0; nr < shape[0]; ++nr)
                _topk_element(data + nr * vdim, k, vdim, ptr + nr * k, sorted, heap_cmp);
        }
    }
}


template<class HeapCmp>
py::array_t<int64_t> topk_element(
        py::array_t<typename HeapCmp::DataType, py::array::c_style | py::array::forcecast> values,
        ssize_t k, bool sorted, ssize_t th_para) {

    if (k == 0) {
        return py::array_t<int64_t>();
    }
    std::vector<int64_t> shape_val;
    arrayshape2vector(shape_val, values);

    std::vector<int64_t> shape_ind(shape_val.size());
    if (shape_ind.size() == 2) {
        shape_ind[0] = shape_val[0];
        shape_ind[1] = k;
    }
    else {
        shape_ind[0] = k;
    }

    std::vector<int64_t> strides;
    shape2strides(shape_ind, strides, (int64_t)0);

    auto result = py::array_t<int64_t>(shape_ind, strides);
    py::buffer_info buf = result.request();
    int64_t* pos = (int64_t*) buf.ptr;
    const typename HeapCmp::DataType* data_val = values.data();
    _topk_element_ptr<HeapCmp>(pos, k, data_val, shape_val, sorted, th_para);
    return result;
}


template<typename NTYPE>
py::array_t<NTYPE> topk_element_fetch(
        py::array_t<NTYPE, py::array::c_style | py::array::forcecast> values,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
    std::vector<int64_t> shape_val;
    arrayshape2vector(shape_val, values);
    
    std::vector<int64_t> shape_ind;
    arrayshape2vector(shape_ind, indices);
    
    auto tdim = flattened_dimension(shape_ind);    
    auto dim_val = shape_val[shape_val.size()-1];
    auto dim_ind = shape_ind[shape_ind.size()-1];
    const NTYPE* data_val = values.data();
    const int64_t* data_ind = indices.data();
    const int64_t* end_ind = data_ind + tdim;
    
    std::vector<int64_t> strides;
    shape2strides(shape_ind, strides, (NTYPE)0);

    auto result = py::array_t<NTYPE>(shape_ind, strides);
    py::buffer_info buf = result.request();
    NTYPE * ptr = (NTYPE*) buf.ptr;
    const int64_t * next_end_ind;
    for(; data_ind != end_ind; data_val += dim_val) {
        next_end_ind = data_ind + dim_ind;
        for( ; data_ind != next_end_ind; ++data_ind, ++ptr)
            *ptr = data_val[*data_ind];
    }
    return result;
}


py::array_t<int64_t> topk_element_min_int64(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> values,
        ssize_t k, bool sorted, ssize_t th_para) {
    return topk_element<HeapMin<int64_t>>(values, k, sorted, th_para);
}


py::array_t<int64_t> topk_element_min_float(
        py::array_t<float, py::array::c_style | py::array::forcecast> values,
        ssize_t k, bool sorted, ssize_t th_para) {
    return topk_element<HeapMin<float>>(values, k, sorted, th_para);
}


py::array_t<int64_t> topk_element_min_double(
        py::array_t<double, py::array::c_style | py::array::forcecast> values,
        ssize_t k, bool sorted, ssize_t th_para) {
    return topk_element<HeapMin<double>>(values, k, sorted, th_para);
}


py::array_t<int64_t> topk_element_max_int64(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> values,
        ssize_t k, bool sorted, ssize_t th_para) {
    return topk_element<HeapMax<int64_t>>(values, k, sorted, th_para);
}


py::array_t<int64_t> topk_element_max_float(
        py::array_t<float, py::array::c_style | py::array::forcecast> values,
        ssize_t k, bool sorted, ssize_t th_para) {
    return topk_element<HeapMax<float>>(values, k, sorted, th_para);
}


py::array_t<int64_t> topk_element_max_double(
        py::array_t<double, py::array::c_style | py::array::forcecast> values,
        ssize_t k, bool sorted, ssize_t th_para) {
    return topk_element<HeapMax<double>>(values, k, sorted, th_para);
}


py::array_t<double> topk_element_fetch_double(
        py::array_t<double, py::array::c_style | py::array::forcecast> values,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
    return topk_element_fetch(values, indices);
}


py::array_t<float> topk_element_fetch_float(
        py::array_t<float, py::array::c_style | py::array::forcecast> values,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
    return topk_element_fetch(values, indices);
}


py::array_t<int64_t> topk_element_fetch_int64(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> values,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
    return topk_element_fetch(values, indices);
}



/////////////////////////////////////////////
// end: topk
/////////////////////////////////////////////




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
            R"pbdoc(C++ implementation of operator ArrayFeatureExtractor for float32.
The function only works with contiguous arrays.)pbdoc");
    m.def("array_feature_extractor_double", &array_feature_extractor_double,
            R"pbdoc(C++ implementation of operator ArrayFeatureExtractor for float64.
The function only works with contiguous arrays.)pbdoc");
    m.def("array_feature_extractor_int64", &array_feature_extractor_int64,
            R"pbdoc(C++ implementation of operator ArrayFeatureExtractor for int64.
The function only works with contiguous arrays.)pbdoc");

    m.def("topk_element_min_float", &topk_element_min_float,
            R"pbdoc(C++ implementation of operator TopK for float32.
The function only works with contiguous arrays.
The function is parallelized for more than *th_para* rows.
It only does it on the last axis.)pbdoc");
    m.def("topk_element_min_double", &topk_element_min_double,
            R"pbdoc(C++ implementation of operator TopK for float32.
The function only works with contiguous arrays.
The function is parallelized for more than *th_para* rows.
It only does it on the last axis.)pbdoc");
    m.def("topk_element_min_int64", &topk_element_min_int64,
            R"pbdoc(C++ implementation of operator TopK for float32.
The function only works with contiguous arrays.
The function is parallelized for more than *th_para* rows.
It only does it on the last axis.)pbdoc");

    m.def("topk_element_max_float", &topk_element_max_float,
            R"pbdoc(C++ implementation of operator TopK for float32.
The function only works with contiguous arrays.
The function is parallelized for more than *th_para* rows.
It only does it on the last axis.)pbdoc");
    m.def("topk_element_max_double", &topk_element_max_double,
            R"pbdoc(C++ implementation of operator TopK for float32.
The function only works with contiguous arrays.
The function is parallelized for more than *th_para* rows.
It only does it on the last axis.)pbdoc");
    m.def("topk_element_max_int64", &topk_element_max_int64,
            R"pbdoc(C++ implementation of operator TopK for float32.
The function only works with contiguous arrays.
The function is parallelized for more than *th_para* rows.
It only does it on the last axis.)pbdoc");

    m.def("topk_element_fetch_float", &topk_element_fetch_float,
            R"pbdoc(Fetches the top k element knowing their indices
on each row (= last dimension for a multi dimension array).)pbdoc");
    m.def("topk_element_fetch_double", &topk_element_fetch_double,
            R"pbdoc(Fetches the top k element knowing their indices
on each row (= last dimension for a multi dimension array).)pbdoc");
    m.def("topk_element_fetch_int64", &topk_element_fetch_int64,
            R"pbdoc(Fetches the top k element knowing their indices
on each row (= last dimension for a multi dimension array).)pbdoc");
}

#endif
