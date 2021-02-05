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
class ConvTranspose : ConvPoolCommon<T> {
    
    private:
        
        AutoPadType auto_pad_;    
        std::vector<int64_t> dilations_;
        int64_t group_;
        std::vector<int64_t> kernel_shape_;
        std::vector<int64_t> pads_;
        std::vector<int64_t> strides_;
        std::vector<int64_t> output_padding_;
        std::vector<int64_t> output_shape_;
    
    public:

        ConvTranspose();
        void init(const std::string &auto_pad,
                  py::array_t<int64_t> dilations,
                  int64_t group,
                  py::array_t<int64_t> kernel_shape,
                  py::array_t<int64_t> pads,
                  py::array_t<int64_t> strides,
                  py::array_t<int64_t> output_padding,
                  py::array_t<int64_t> output_shape);

        py::array_t<T> compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                               py::array_t<T, py::array::c_style | py::array::forcecast> W,
                               py::array_t<T, py::array::c_style | py::array::forcecast> B) const;
    
    private:

        void compute_kernel_shape(const std::vector<int64_t>& weight_shape,
                                  std::vector<int64_t>& kernel_shape) const;

        void compute_gil_free(py::array_t<T> X, py::array_t<T> W,
                              py::array_t<T> B, py::array_t<T>& Y,
                              const std::vector<int64_t>& input_shape,
                              const std::vector<int64_t>& output_shape,
                              const std::vector<int64_t>& kernel_shape,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& x_dims,
                              const std::vector<int64_t>& y_dims,
                              const std::vector<int64_t>& w_dims) const;

        void infer_output_shape(const std::vector<int64_t>& x_dims,
                                const std::vector<int64_t>& w_dims,
                                const std::vector<int64_t>& input_shape,
                                const std::vector<int64_t>& kernel_shape,
                                const std::vector<int64_t>& strides_p,
                                const std::vector<int64_t>& dilations_p,
                                std::vector<int64_t>& pads_p,
                                std::vector<int64_t>& output_shape,
                                const std::vector<int64_t>& output_padding,
                                AutoPadType auto_pad,
                                bool ForceSymmetricAutoPadding = false) const;
};

template<typename T>
ConvTranspose<T>::ConvTranspose() : ConvPoolCommon<T>() {
}


template<typename T>
void ConvTranspose<T>::init(
            const std::string &auto_pad,
            py::array_t<int64_t> dilations,
            int64_t group,
            py::array_t<int64_t> kernel_shape,
            py::array_t<int64_t> pads,
            py::array_t<int64_t> strides,
            py::array_t<int64_t> output_padding,
            py::array_t<int64_t> output_shape
    ) {
    auto_pad_ = to_AutoPadType(auto_pad);
    array2vector(dilations_, dilations, int64_t);
    group_ = group;        
    array2vector(dilations_, dilations, int64_t);
    array2vector(pads_, pads, int64_t);
    array2vector(strides_, strides, int64_t);
    array2vector(output_padding_, output_padding, int64_t);
    array2vector(output_shape_, output_shape, int64_t);
}


template<typename T>
void ConvTranspose<T>::compute_kernel_shape(const std::vector<int64_t>& weight_shape,
                                            std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_.size() > 0) {
        kernel_shape = kernel_shape_;
        if (kernel_shape.size() + 2 != weight_shape.size())
            throw std::runtime_error(
                "kernel_shape num_dims is not compatible with W num_dims (1).");

        for (size_t i = 0; i < kernel_shape.size(); ++i)
            if (kernel_shape[i] != weight_shape[i + 2])
                throw std::runtime_error(
                    "kernel_shape num_dims is not compatible with W num_dims (2).");
    }
    else {
        auto& weight_dims = weight_shape;
        kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }
}


template<typename T>
py::array_t<T> ConvTranspose<T>::compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                                         py::array_t<T, py::array::c_style | py::array::forcecast> W,
                                         py::array_t<T, py::array::c_style | py::array::forcecast> B) const {

    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    std::vector<int64_t> w_dims;
    arrayshape2vector(w_dims, W);

    // const int64_t N = x_dims[0];
    // const int64_t C = x_dims[1];
    // const int64_t M = w_dims[1];

    std::vector<int64_t> input_shape(x_dims.begin() + 2, x_dims.end());

    std::vector<int64_t> kernel_shape;
    compute_kernel_shape(w_dims, kernel_shape);

    std::vector<int64_t> pads;
    pads.reserve(input_shape.size() * 2);
    pads.assign(pads_.begin(), pads_.end());
    if (pads.empty())
        pads.resize(kernel_shape.size() * 2, 0);

    std::vector<int64_t> dilations(dilations_);
    if (dilations.empty())
        dilations.resize(kernel_shape.size(), 1);

    std::vector<int64_t> strides(strides_);
    if (strides.empty())
        strides.resize(kernel_shape.size(), 1);

    std::vector<int64_t> output_padding(output_padding_);
    if (output_padding.empty())
        output_padding.resize(kernel_shape.size(), 0);

    std::vector<int64_t> y_dims;
    infer_output_shape(x_dims, w_dims, input_shape, kernel_shape, strides, dilations,
                       pads, y_dims, output_padding, auto_pad_);
    std::vector<int64_t> output_shape(y_dims.begin() + 2, y_dims.end());

    py::array_t<T> Y(y_dims);
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
void ConvTranspose<T>::infer_output_shape(
                    const std::vector<int64_t>& x_dims,
                    const std::vector<int64_t>& w_dims,
                    const std::vector<int64_t>& input_shape,
                    const std::vector<int64_t>& kernel_shape,
                    const std::vector<int64_t>& strides_p,
                    const std::vector<int64_t>& dilations_p,
                    std::vector<int64_t>& pads_p,
                    std::vector<int64_t>& output_shape,
                    const std::vector<int64_t>& output_padding,
                    AutoPadType auto_pad,
                    bool ForceSymmetricAutoPadding) const {

    size_t output_shape_size = output_shape.size();
    output_shape.insert(output_shape.begin(), {x_dims[0], w_dims[1] * group_});

    size_t rank = input_shape.size();
    if (rank > strides_p.size())
        throw std::runtime_error("rank out of 'strides_p' boundary.");
    if (rank > kernel_shape.size())
        throw std::runtime_error("rank out of 'kernel_shape' boundary.");
    if (rank > dilations_p.size())
        throw std::runtime_error("rank out of 'dilations_p' boundary.");
    if (rank > output_padding.size())
        throw std::runtime_error("rank out of 'output_padding' boundary.");
    if (rank > pads_p.size())
        throw std::runtime_error("rank out of 'pads_p' boundary.");

    for (size_t dim = 0; dim < rank; ++dim) {
        int64_t dim_size = -1;

        if (output_shape_size != 0)
            dim_size = output_shape_size == rank ? output_shape[dim] : output_shape[dim + 2];

        ComputeTransposePadAndOutputShape<T>(
            input_shape[dim],
            strides_p[dim],
            kernel_shape[dim],
            dilations_p[dim],
            output_padding[dim],
            auto_pad,
            &pads_p.at(dim),
            &pads_p.at(input_shape.size() + dim),
            &dim_size);

        if (dim_size <= 0)
            throw std::runtime_error("Invalid argument in infer_output_shape.");
        output_shape.push_back(dim_size);
    }
}


template<typename T>
void ConvTranspose<T>::compute_gil_free(
        py::array_t<T> X, py::array_t<T> W, py::array_t<T> B, py::array_t<T>& Y,
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
    // const int64_t num_input_channels = C;
    const int64_t num_output_channels = w_dims[1] * group_;
    const int64_t M = w_dims[0];

    const int64_t input_shape_size = flattened_dimension(input_shape);
    const int64_t output_image_size = flattened_dimension(y_dims, 2);
    // const int64_t output_shape_size = flattened_dimension(output_shape);
    const int64_t y_size = flattened_dimension(y_dims);
    const int64_t kernel_size = flattened_dimension(kernel_shape);
    const int64_t X_offset = C / group_ * input_shape_size;
    const int64_t Y_offset = flattened_dimension(y_dims) / y_dims[0] / group_;
    const int64_t W_offset = flattened_dimension(w_dims) / group_;
    const int64_t kernel_dim = num_output_channels / group_ * kernel_size;

    std::vector<int64_t> col_buffer_shape{kernel_dim};
    col_buffer_shape.insert(col_buffer_shape.end(), input_shape.begin(),
                            input_shape.end());
    int64_t col_buffer_size = flattened_dimension(col_buffer_shape);


    std::vector<T> _col_data(col_buffer_size);
    auto col_buffer_data = &_col_data[0];
 
    const T* Xdata = X.data(0);
    T* Ydata = (T*)Y.data(0);
    T* yptr;
    size_t k2;

    std::fill(Ydata, Ydata + y_size, (T)0);

    // const size_t kernel_rank = kernel_shape.size();
    
    std::vector<int64_t> output_shape2(y_dims.begin() + 1, y_dims.end());
    output_shape2[0] /= group_;
    const int64_t output_shape2_size = flattened_dimension(output_shape2);

    for (int image_id = 0; image_id < N; ++image_id) {
        for (int group_id = 0; group_id < group_; ++group_id) {

            gemm<T>(
                true,
                false,
                (size_t)kernel_dim,  // m
                (size_t)(input_shape_size),  // n
                (size_t)(C / group_),  // k
                (T)1, // alpha
                (const T*)W.data(0) + group_id * W_offset, // *a
                (const T*)Xdata + group_id * X_offset, // *b
                (T)0,  // beta
                col_buffer_data // *c
            );

            Im2colNd_NCHW<T>(
                col_buffer_data,
                &output_shape2[0],
                col_buffer_shape.data(),
                output_shape2_size,
                col_buffer_size,
                &kernel_shape[0],
                strides.data(),
                &dilations[0],
                &pads[0],
                static_cast<int>(kernel_shape.size()),
                (T*)Ydata + group_id * Y_offset,
                true);
        }
            
        if (b_dims.size() != 0 && b_dims[0] != 0) {
            // conv: output_image_size, M
            // convt: output_image_size, num_output_channels
            const T* ptrb = B.data(0);
            for(size_t k = 0; k < (size_t)num_output_channels; ++k, ++ptrb) {
                yptr = Ydata + output_image_size * k;
                for(k2 = 0; k2 < (size_t)output_image_size; ++k2, ++yptr)
                    *yptr += *ptrb;
            }
        }

        Xdata += X_offset * group_;
        Ydata += Y_offset * group_;
    }
}


class ConvTransposeFloat : public ConvTranspose<float> {
    public:
        ConvTransposeFloat() : ConvTranspose<float>() {}
};


class ConvTransposeDouble : public ConvTranspose<double> {
    public:
        ConvTransposeDouble() : ConvTranspose<double>() {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_conv_transpose_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements ConvTranspose operator."
    #else
    R"pbdoc(Implements runtime for operator Conv. The code is inspired from
`conv_transpose.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv_transpose.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<ConvTransposeFloat> clf (m, "ConvTransposeFloat",
        R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`conv_transpose.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv_transpose.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &ConvTransposeFloat::init,
            "Initializes the runtime with the ONNX attributes.");
    clf.def("compute", &ConvTransposeFloat::compute,
            "Computes the output for operator ConvTranspose.");

    py::class_<ConvTransposeDouble> cld (m, "ConvTransposeDouble",
        R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`conv_transpose.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv_transpose.cc>`_
in :epkg:`onnxruntime`. Supports double only.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &ConvTransposeDouble::init,
            "Initializes the runtime with the ONNX attributes.");
    cld.def("compute", &ConvTransposeDouble::compute,
            "Computes the output for operator ConvTranspose.");
}

#endif
