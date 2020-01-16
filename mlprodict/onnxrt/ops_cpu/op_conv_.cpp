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

#include "op_common_.hpp"


template <typename T>
class Conv {
    
    private:
        
        AutoPadType auto_pad_;    
        std::vector<int64_t> dilations_;
        int64_t group_;
        std::vector<int64_t> kernel_shape_;
        std::vector<int64_t> pads_;
        std::vector<int64_t> strides_;
    
    public:

        Conv();
        void init(
            const std::string &auto_pad,
            py::array_t<int64_t> dilations,
            int64_t group,
            py::array_t<int64_t> kernel_shape,
            py::array_t<int64_t> pads,
            py::array_t<int64_t> strides
        );

        py::array_t<T> compute(py::array_t<T> X, py::array_t<T> W, py::array_t<T> B) const;
    
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

        void Im2colNd_NCHW(
            const T* data_img, const int64_t* im_shape,
            const int64_t* col_shape, int64_t /*img_size*/,
            int64_t /*col_size*/, const int64_t* kernel_shape,
            const int64_t* stride, const int64_t* dilation,
            const int64_t* pad, int64_t N, T* data_col, bool accumulate_output = false,
            T padding_value = 0) const;
    
        void Im2col_NCHW(
            const T* data_im, int64_t channels, int64_t height,
            int64_t width, int64_t kernel_h, int64_t kernel_w,
            int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
            int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
            int64_t stride_w, T* data_col, T padding_value = 0) const;
    
        void infer_output_shape(
                const std::vector<int64_t>& input_shape,
                const std::vector<int64_t>& kernel_shape,
                const std::vector<int64_t>& strides_p,
                const std::vector<int64_t>& dilations_p,
                std::vector<int64_t>& pads_p,
                std::vector<int64_t>& output_shape,
                bool ForceSymmetricAutoPadding=false) const;
                
        void ComputePadAndOutputShape(
                const int64_t in_dim, const int64_t stride,
                const int64_t kernel, const int64_t dilation,
                AutoPadType pad_type, int64_t* pad_head,
                int64_t* pad_tail, int64_t* out_dim,
                bool ForceSymmetricAutoPadding = false) const;
};

template<typename T>
Conv<T>::Conv() {
}


template<typename T>
void Conv<T>::init(
            const std::string &auto_pad,
            py::array_t<int64_t> dilations,
            int64_t group,
            py::array_t<int64_t> kernel_shape,
            py::array_t<int64_t> pads,
            py::array_t<int64_t> strides
    ) {
    auto_pad_ = to_AutoPadType(auto_pad);
    array2vector(dilations_, dilations, int64_t);
    group_ = group;        
    array2vector(dilations_, dilations, int64_t);
    array2vector(pads_, pads, int64_t);
    array2vector(strides_, strides, int64_t);
}


template <typename T>
void Conv<T>::Im2colNd_NCHW (
        const T* data_img, const int64_t* im_shape,
        const int64_t* col_shape, int64_t /*img_size*/,
        int64_t /*col_size*/, const int64_t* kernel_shape,
        const int64_t* stride, const int64_t* dilation,
        const int64_t* pad, int64_t N, T* data_col, bool accumulate_output,
        T padding_value) const {
    int64_t kernel_size = 1;
    for (int64_t i = 0; i < N; ++i)
        kernel_size *= kernel_shape[i];

    int64_t channels_col = col_shape[0];
    std::vector<int64_t> d_offset(N, 0);
    std::vector<int64_t> d_iter(N, 0);
    for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
        // Loop over spatial axes in reverse order to compute a per-axis offset.
        int64_t offset = c_col;
        for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
            if (d_i < N - 1)
                offset /= kernel_shape[d_i + 1];
            d_offset[d_i] = offset % kernel_shape[d_i];
        }
        for (bool incremented = true; incremented;) {
            // Loop over spatial axes in forward order to compute the indices in the
            // image and column, and whether the index lies in the padding.
            int64_t index_col = c_col;
            int64_t index_im = c_col / kernel_size;
            bool is_padding = false;
            for (int64_t d_i = 0; d_i < N; ++d_i) {
                int64_t d = d_iter[d_i];
                int64_t d_im = d * stride[d_i] - pad[d_i] + d_offset[d_i] * dilation[d_i];
                is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
                index_col *= col_shape[d_i + 1];
                index_col += d;
                index_im *= im_shape[d_i + 1];
                index_im += d_im;
            }
            if (!accumulate_output) {
                if (is_padding)
                    data_col[index_col] = padding_value;
                else
                    data_col[index_col] = data_img[index_im];
            } 
            else if (!is_padding)   // col2im
                data_col[index_im] += data_img[index_col];

            // Loop over spatial axes in reverse order to choose an index,
            // like counting.
            incremented = false;
            for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
                int64_t d_max = col_shape[d_i + 1];
                // ORT_ENFORCE(d_iter[d_i] < d_max);
                if (d_iter[d_i] == d_max - 1)
                    d_iter[d_i] = 0;
                else {  // d_iter[d_i] < d_max - 1
                    ++d_iter[d_i];
                    incremented = true;
                    break;
                }
            }
        }  // while(incremented) {
    }    // for (int c = 0; c < channels_col; ++c) {
}


template<typename T>
void Conv<T>::Im2col_NCHW(
        const T* data_im, int64_t channels, int64_t height,
        int64_t width, int64_t kernel_h, int64_t kernel_w,
        int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
        int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
        int64_t stride_w, T* data_col, T padding_value) const {
    const int64_t output_h =
        (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
        1;
    const int64_t output_w =
        (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
        1;
  
    // Fast path for zero padding and no dilation
    // From Torch, THNN_(unfolded_copy)
    if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
            pad_t == 0 && pad_b == 0) {
        for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
            const auto nip = k / (kernel_h * kernel_w);
            const auto rest = k % (kernel_h * kernel_w);
            const auto kh = rest / kernel_w;
            const auto kw = rest % kernel_w;
            auto* dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) +
                        kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
            const auto* src = data_im + nip * (height * width);
            for (auto y = 0; y < output_h; y++) {
                const auto iy = y * stride_h + kh;
                const auto ix = kw;
                if (stride_w == 1) {
                    memcpy(
                        dst + (y * output_w),
                        src + (iy * width + ix),
                        sizeof(T) * output_w);
                } 
                else {
                    for (auto x = 0; x < output_w; x++) {
                        memcpy(
                            dst + (y * output_w + x),
                            src + (iy * width + ix + x * stride_w),
                            sizeof(T));
                    }
                }
            }
        }
        return;
    }
  
    // Fast path for equal padding
    if (pad_l == pad_r && pad_t == pad_b) {
        Im2colWithEqualPadding(output_h, output_w, data_im, channels, height, width, kernel_h, kernel_w, dilation_h, dilation_w, pad_t, pad_l, stride_h, stride_w, data_col, padding_value);
        return;
    }
  
    // Baseline
    const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
    const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;
  
    int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
    int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  
    int64_t channels_col = channels * kernel_h * kernel_w;
    for (int64_t c = 0; c < channels_col; ++c) {
        int64_t w_offset = c % kernel_w;
        int64_t h_offset = (c / kernel_w) % kernel_h;
        int64_t c_im = c / kernel_h / kernel_w;
        for (int64_t h = 0; h < height_col; ++h) {
            for (int64_t w = 0; w < width_col; ++w) {
                int64_t h_pad = h * stride_h - pad_t + h_offset * dilation_h;
                int64_t w_pad = w * stride_w - pad_l + w_offset * dilation_w;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_col[(c * height_col + h) * width_col + w] =
                        data_im[(c_im * height + h_pad) * width + w_pad];
                else
                    data_col[(c * height_col + h) * width_col + w] = padding_value;
            }
        }
    }
}


template<typename T>
void Conv<T>::compute_kernel_shape(const std::vector<int64_t>& weight_shape,
                                   std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_.size() > 0) {
        kernel_shape = kernel_shape_;
        if (kernel_shape.size() + 2 != weight_shape.size())
            throw std::runtime_error("kernel_shape num_dims is not compatible with W num_dims (1).");

        for (size_t i = 0; i < kernel_shape.size(); ++i)
            if (kernel_shape[i] != weight_shape[i + 2])
                throw std::runtime_error("kernel_shape num_dims is not compatible with W num_dims (2).");
    }
    else {
        auto& weight_dims = weight_shape;
        kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }
}


template<typename T>
py::array_t<T> Conv<T>::compute(py::array_t<T> X, py::array_t<T> W, py::array_t<T> B) const {
    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    std::vector<int64_t> w_dims;
    arrayshape2vector(w_dims, W);
    
    const int64_t N = x_dims[0];
    const int64_t C = x_dims[1];
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
    infer_output_shape(input_shape, kernel_shape, strides, dilations, pads, y_dims);
    std::vector<int64_t> output_shape(y_dims.begin() + 2, y_dims.end());
    
    py::array_t<T> Y(flattened_dimension(output_shape));
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
void Conv<T>::infer_output_shape(const std::vector<int64_t>& input_shape,
              const std::vector<int64_t>& kernel_shape,
              const std::vector<int64_t>& strides_p,
              const std::vector<int64_t>& dilations_p,
              std::vector<int64_t>& pads_p,
              std::vector<int64_t>& output_shape,
              bool ForceSymmetricAutoPadding) const {

    size_t rank = input_shape.size();
    int64_t dim_size;
                  
    for (size_t dim = 0; dim < rank; ++dim) {
        if (dim >= strides_p.size() || dim >= kernel_shape.size() ||
                dim >= dilations_p.size() || dim >= pads_p.size() ||
                rank + dim >= pads_p.size())
            throw std::runtime_error("Failure.");
        
        dim_size = 0;
        ComputePadAndOutputShape(
            input_shape[dim], strides_p[dim], kernel_shape[dim],
            dilations_p[dim], auto_pad_, &pads_p.at(dim),
            &pads_p.at(input_shape.size() + dim),
            &dim_size, ForceSymmetricAutoPadding);
        if (dim_size <= 0)
            throw std::runtime_error("Invalid argument.");
        output_shape.push_back(dim_size);
    }
}

template <typename T>
void Conv<T>::ComputePadAndOutputShape(
        const int64_t in_dim, const int64_t stride,
        const int64_t kernel, const int64_t dilation,
        AutoPadType pad_type, int64_t* pad_head,
        int64_t* pad_tail, int64_t* out_dim,
        bool ForceSymmetricAutoPadding) const {

    const int64_t dkernel = dilation * (kernel - 1) + 1;

    if (pad_type == AutoPadType::NOTSET) {
        *out_dim = static_cast<int64_t>(static_cast<float>(
            in_dim + *pad_head + *pad_tail - dkernel) / stride + 1);
    }
    else {
        switch (pad_type) {
            case AutoPadType::VALID:
                *pad_head = 0;
                *pad_tail = 0;
                *out_dim = (in_dim - dkernel) / stride + 1;
                break;
            case AutoPadType::SAME_UPPER:
            case AutoPadType::SAME_LOWER: {
                if (dilation != 1)
                    throw std::runtime_error("Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");
                int64_t legacy_target_size = (in_dim + stride - 1) / stride;
                int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
                *out_dim = (in_dim + pad_needed - dkernel) / stride + 1;

                // make sure padding is symmetric
                if (ForceSymmetricAutoPadding)
                    pad_needed = roundUpPow2<int64_t, 2>(pad_needed);

                if (pad_type == AutoPadType::SAME_LOWER)
                    *pad_head = (pad_needed + 1) / 2;
                else
                    *pad_head = pad_needed / 2;
                *pad_tail = pad_needed - *pad_head;
                } break;
            default:
                throw std::runtime_error("Invalid argument.");
        }
    }
}


template<typename T>
void Conv<T>::compute_gil_free(
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

    const int64_t N = x_dims[0];
    const int64_t C = x_dims[1];
    const int64_t M = w_dims[0];

    const int64_t input_image_size = input_shape.size();
    const int64_t output_image_size = output_shape.size();
    const int64_t kernel_size = kernel_shape.size();
    const int64_t X_offset = C / group_ * input_image_size;
    const int64_t Y_offset = y_dims.size() / y_dims[0] / group_;
    const int64_t W_offset = w_dims.size() / group_;
    const int64_t kernel_dim = C / group_ * kernel_size;
    const int64_t col_buffer_size = kernel_dim * output_image_size;

    std::vector<T> _col_data(col_buffer_size);
    auto col_buffer_data = &_col_data[0];

    const T* Xdata = X.data(0);
    const T* Ydata = Y.data(0);

    std::vector<int64_t> image_shape(x_dims.begin() + 1, x_dims.end());
    std::vector<int64_t> col_buffer_shape{kernel_dim};
    col_buffer_shape.insert(col_buffer_shape.end(), output_shape.begin(),
                            output_shape.end());

    const size_t kernel_rank = kernel_shape.size();

    for (int image_id = 0; image_id < N; ++image_id) {
        for (int group_id = 0; group_id < group_; ++group_id) {
            if (kernel_rank == 2) {
                Im2col_NCHW(
                    Xdata + group_id * X_offset,
                    C / group_,
                    input_shape[0],
                    input_shape[1],
                    kernel_shape[0],
                    kernel_shape[1],
                    dilations[0],
                    dilations[1],
                    pads[0],
                    pads[1],
                    pads[2],
                    pads[3],
                    strides[0],
                    strides[1],
                    col_buffer_data);
            }
            else {
                Im2colNd_NCHW(
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
            cblas_gemm(
                CblasNoTrans,
                CblasNoTrans,
                M / group_,  // m
                output_image_size,  // n
                kernel_dim,  // k
                1, // alpha
                W.data(0) + group_id * W_offset, // *a
                col_buffer_data, // *b
                0,  // beta
                Ydata + group_id * Y_offset // *c);
        }

        if (B != nullptr) {
            auto Ymatrix = EigenMatrixMap<T>(Ydata, output_image_size, M);
            auto Bvec = ConstEigenVectorMap<T>(B.data(0), M);
            Ymatrix.rowwise() += Bvec.transpose();
        }

        Xdata += X_offset * conv_attrs_.group;
        Ydata += Y_offset * conv_attrs_.group;
    }
}


class ConvFloat : public Conv<float>
{
    public:
        ConvFloat() : Conv<float>() {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_tree_ensemble_classifier_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements Conv operator."
    #else
    R"pbdoc(Implements runtime for operator Conv. The code is inspired from
`tree_ensemble_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;
    
    py::class_<ConvFloat> clf (m, "ConvFloat",
        R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &ConvFloat::init,
            "Initializes the runtime with the ONNX attributes.");
    clf.def("compute", &ConvFloat::compute,
            "Computes the output for operator Conv.");    
}

#endif
