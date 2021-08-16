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

#if USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
#endif

#include "op_conv_matrices_.hpp"


template <typename T>
class Conv : public ConvPoolCommon {
    
    public:

        Conv();

        py::array_t<T> compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                               py::array_t<T, py::array::c_style | py::array::forcecast> W,
                               py::array_t<T, py::array::c_style | py::array::forcecast> B) const;
    
    protected:

        void compute_gil_free(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                              py::array_t<T, py::array::c_style | py::array::forcecast> W,
                              py::array_t<T, py::array::c_style | py::array::forcecast> B,
                              py::array_t<T, py::array::c_style | py::array::forcecast>& Y,
                              const std::vector<int64_t>& input_shape,
                              const std::vector<int64_t>& output_shape,
                              const std::vector<int64_t>& kernel_shape,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& x_dims,
                              const std::vector<int64_t>& y_dims,
                              const std::vector<int64_t>& w_dims) const;
};

template<typename T>
Conv<T>::Conv() : ConvPoolCommon() {
}


template<typename T>
py::array_t<T> Conv<T>::compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                                py::array_t<T, py::array::c_style | py::array::forcecast> W,
                                py::array_t<T, py::array::c_style | py::array::forcecast> B) const {

    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    std::vector<int64_t> w_dims;
    arrayshape2vector(w_dims, W);

    const int64_t N = x_dims[0];
    // const int64_t C = x_dims[1];
    const int64_t M = w_dims[0];

    std::vector<int64_t> kernel_shape;
    compute_kernel_shape(w_dims, kernel_shape);

    std::vector<int64_t> pads(pads_);
    if (pads.empty())
        pads.resize(kernel_shape.size() * 2, 0);

    std::vector<int64_t> dilations(dilations_);
    if (dilations.empty())
        dilations.resize(kernel_shape.size(), 1);

    std::vector<int64_t> strides(strides_);
    if (strides.empty())
        strides.resize(kernel_shape.size(), 1);

    std::vector<int64_t> y_dims;
    y_dims.insert(y_dims.begin(), {N, M});
    std::vector<int64_t> input_shape(x_dims.begin() + 2, x_dims.end());
    infer_output_shape(input_shape, kernel_shape, strides, dilations, pads, y_dims, false);
    std::vector<int64_t> output_shape(y_dims.begin() + 2, y_dims.end());

    // py::array::ShapeContainer shape(y_dims);
    // auto total_size = flattened_dimension(y_dims);
    py::array_t<T, py::array::c_style | py::array::forcecast> Y(y_dims);
    {
        py::gil_scoped_release release;
        compute_gil_free(X, W, B, Y,
                         input_shape, output_shape,
                         kernel_shape, pads, dilations, strides,
                         x_dims, y_dims, w_dims);
    }
    return Y;
}


template<typename T>
void Conv<T>::compute_gil_free(
        py::array_t<T, py::array::c_style | py::array::forcecast> X,
        py::array_t<T, py::array::c_style | py::array::forcecast> W,
        py::array_t<T, py::array::c_style | py::array::forcecast> B,
        py::array_t<T, py::array::c_style | py::array::forcecast>& Y,
        const std::vector<int64_t>& input_shape,
        const std::vector<int64_t>& output_shape,
        const std::vector<int64_t>& kernel_shape,
        const std::vector<int64_t>& pads,
        const std::vector<int64_t>& dilations,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& x_dims,
        const std::vector<int64_t>& y_dims,
        const std::vector<int64_t>& w_dims
        ) const {

    std::vector<int64_t> b_dims;
    arrayshape2vector(b_dims, B);
            
    const int64_t N = x_dims[0];
    const int64_t C = x_dims[1];
    const int64_t M = w_dims[0];
            
    const int64_t input_image_size = flattened_dimension(input_shape);
    const int64_t output_image_size = flattened_dimension(output_shape);
    const int64_t y_size = flattened_dimension(y_dims);
    const int64_t kernel_size = flattened_dimension(kernel_shape);
    const int64_t X_offset = C / group_ * input_image_size;
    const int64_t Y_offset = flattened_dimension(y_dims) / y_dims[0] / group_;
    const int64_t W_offset = flattened_dimension(w_dims) / group_;
    const int64_t kernel_dim = C / group_ * kernel_size;
    const int64_t col_buffer_size = kernel_dim * output_image_size;

    std::vector<T> _col_data(col_buffer_size);
    auto col_buffer_data = &_col_data[0];
 
    const T* Xdata = X.data(0);
    T* Ydata = (T*)Y.data(0);
    T* yptr;
    size_t k2;

    std::fill(Ydata, Ydata + y_size, (T)0);

    std::vector<int64_t> image_shape(x_dims.begin() + 1, x_dims.end());
    std::vector<int64_t> col_buffer_shape{kernel_dim};
    col_buffer_shape.insert(col_buffer_shape.end(), output_shape.begin(),
                            output_shape.end());

    const size_t kernel_rank = kernel_shape.size();

    for (int image_id = 0; image_id < N; ++image_id) {
        for (int group_id = 0; group_id < group_; ++group_id) {
            if (kernel_rank == 2) {
                Im2col_NCHW<T>(
                    Xdata + group_id * X_offset,
                    C / group_,
                    input_shape[0], input_shape[1],
                    kernel_shape[0], kernel_shape[1],
                    dilations[0], dilations[1],
                    pads[0], pads[1], pads[2], pads[3],
                    strides[0], strides[1],
                    col_buffer_data);
            }
            else {
                Im2colNd_NCHW<T>(
                    Xdata + group_id * X_offset,
                    &image_shape[0],
                    col_buffer_shape.data(),
                    C * input_image_size,
                    col_buffer_size,
                    &kernel_shape[0],
                    strides.data(),
                    &dilations[0],
                    &pads[0],
                    static_cast<int>(kernel_shape.size()),
                    col_buffer_data);
            }
           
            // C := alpha*op(A)*op(B) + beta*C
            // void cblas_sgemm (const CBLAS_LAYOUT Layout,
            //              const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
            //              const MKL_INT m, const MKL_INT n, const MKL_INT k,
            //              const float alpha, const float *a, const MKL_INT lda,
            //              const float *b, const MKL_INT ldb, const float beta,
            //              float *c, const MKL_INT ldc);
            gemm<T>(
                false,
                false,
                (size_t)(M / group_),  // m
                (size_t)(output_image_size),  // n
                (size_t)kernel_dim,  // k
                (T)1, // alpha
                (const T*)W.data(0) + group_id * W_offset, // *a
                (const T*)col_buffer_data, // *b
                (T)0,  // beta
                (T*)Ydata + group_id * Y_offset // *c
            );
        }

        if (b_dims.size() != 0 && b_dims[0] != 0) {
            const T* ptrb = B.data(0);
            for(size_t k = 0; k < (size_t)M; ++k, ++ptrb) {
                yptr = Ydata + output_image_size * k;
                for(k2 = 0; k2 < (size_t)output_image_size; ++k2, ++yptr)
                    *yptr += *ptrb;
            }
        }

        Xdata += X_offset * group_;
        Ydata += Y_offset * group_;
    }    
}


class ConvFloat : public Conv<float> {
    public:
        ConvFloat() : Conv<float>() {}
};


class ConvDouble : public Conv<double> {
    public:
        ConvDouble() : Conv<double>() {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_conv_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements Conv operator."
    #else
    R"pbdoc(Implements runtime for operator Conv. The code is inspired from
`conv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<ConvFloat> clf (m, "ConvFloat",
        R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`conv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &ConvFloat::init,
            "Initializes the runtime with the ONNX attributes.");
    clf.def("compute", &ConvFloat::compute,
            "Computes the output for operator Conv.");

    py::class_<ConvDouble> cld (m, "ConvDouble",
        R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`conv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv.cc>`_
in :epkg:`onnxruntime`. Supports double only.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &ConvDouble::init,
            "Initializes the runtime with the ONNX attributes.");
    cld.def("compute", &ConvDouble::compute,
            "Computes the output for operator Conv.");
}

#endif
