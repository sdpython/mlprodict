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
class MaxPool : ConvPoolCommon<T> {
    
    private:
        
        AutoPadType auto_pad_;
        int64_t ceil_mode_;
        int64_t storage_order_;
        std::vector<int64_t> pads_;
        std::vector<int64_t> strides_;
        std::vector<int64_t> kernel_shape_;
        std::vector<int64_t> dilations_;
    
        bool global_pooling_;
    
    public:

        MaxPool();
        void init(const std::string &auto_pad,
                  py::array_t<int64_t> dilations,
                  int64_t ceil_mode,
                  int64_t storage_order,
                  py::array_t<int64_t> kernel_shape,
                  py::array_t<int64_t> pads,
                  py::array_t<int64_t> strides);

        py::tuple compute(py::array_t<T, py::array::c_style | py::array::forcecast> X) const;
    
    private:

        std::vector<int64_t> SetOutputSize(const std::vector<int64_t>& input_shape,
                                           int64_t output_channel,
                                           std::vector<int64_t>* actual_pads,
                                           std::vector<int64_t>* actual_strides,
                                           std::vector<int64_t>* actual_kernel_shape,
                                           std::vector<int64_t>* actual_dilations) const;

        void InferOutputSize(const std::vector<int64_t>& input_dims,
                             std::vector<int64_t>* output_dims,
                             std::vector<int64_t>* actual_pads,
                             std::vector<int64_t>* actual_strides,
                             std::vector<int64_t>* actual_kernel_shape,
                             std::vector<int64_t>* actual_dilations) const;
    
        void ComputeSizePadDilations(const int64_t in_size,
                                     const int64_t stride,
                                     const int64_t kernel,
                                     int64_t* pad_head,
                                     int64_t* pad_tail,
                                     int64_t dilation,
                                     int64_t* out_size) const;
    
        int64_t ComputeOutputSize(int64_t in_size,
                                  int64_t stride,
                                  int64_t kernel,
                                  int64_t pad_needed,
                                  int64_t dilation) const;

        void compute_gil_free(py::array_t<T> X, py::array_t<T>& Y, py::array_t<int64_t>* I,
                              const std::vector<int64_t>& kernel_shape,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& x_dims,
                              const std::vector<int64_t>& y_dims) const;
};


template<typename T>
MaxPool<T>::MaxPool() : ConvPoolCommon<T>() {
    global_pooling_ = false;
}


template<typename T>
void MaxPool<T>::init(
            const std::string &auto_pad,
            py::array_t<int64_t> dilations,
            int64_t ceil_mode,
            int64_t storage_order,
            py::array_t<int64_t> kernel_shape,
            py::array_t<int64_t> pads,
            py::array_t<int64_t> strides
    ) {
    auto_pad_ = to_AutoPadType(auto_pad);
    array2vector(dilations_, dilations, int64_t);
    ceil_mode_ = ceil_mode;        
    storage_order_ = storage_order;        
    array2vector(dilations_, dilations, int64_t);
    array2vector(pads_, pads, int64_t);
    array2vector(strides_, strides, int64_t);
    array2vector(kernel_shape_, kernel_shape, int64_t);
}


template<typename T>
std::vector<int64_t> MaxPool<T>::SetOutputSize(const std::vector<int64_t>& input_shape,
                                               int64_t output_channel,
                                               std::vector<int64_t>* actual_pads,
                                               std::vector<int64_t>* actual_strides,
                                               std::vector<int64_t>* actual_kernel_shape,
                                               std::vector<int64_t>* actual_dilations) const {
    std::vector<int64_t> output_dims;
    int64_t N = input_shape[0];
    InferOutputSize(input_shape, &output_dims, actual_pads, actual_strides,
                    actual_kernel_shape, actual_dilations);
    output_dims.insert(output_dims.begin(), {N, output_channel});
    return output_dims;
}


template<typename T>
void MaxPool<T>::InferOutputSize(const std::vector<int64_t>& input_dims,
                                 std::vector<int64_t>* output_dims,
                                 std::vector<int64_t>* actual_pads,
                                 std::vector<int64_t>* actual_strides,
                                 std::vector<int64_t>* actual_kernel_shape,
                                 std::vector<int64_t>* actual_dilations) const {
    if (global_pooling_) {
        output_dims->assign(input_dims.size() - 2, 1);
    }
    else {
        for (size_t dim = 0; dim < input_dims.size() - 2; ++dim) {
            int64_t dim_size = 0;
            ComputeSizePadDilations(input_dims[dim + 2],
                                    actual_strides->at(dim),
                                    actual_kernel_shape->at(dim),
                                    &actual_pads->at(dim),
                                    &actual_pads->at(input_dims.size() + dim - 2),
                                    actual_dilations->at(dim),
                                    &dim_size);
            output_dims->push_back(dim_size);
        }
    }
}


template<typename T>
void MaxPool<T>::ComputeSizePadDilations(const int64_t in_size,
                                         const int64_t stride,
                                         const int64_t kernel,
                                         int64_t* pad_head,
                                         int64_t* pad_tail,
                                         int64_t dilation,
                                         int64_t* out_size) const {
    if (auto_pad_ != AutoPadType::NOTSET) {
        switch (auto_pad_) {
            case AutoPadType::VALID:
                *pad_head = 0;
                *pad_tail = 0;
                *out_size = ComputeOutputSize(in_size, stride, kernel, 0, dilation);
                break;
            case AutoPadType::SAME_LOWER: {
                int64_t legacy_target_size = (in_size + stride - 1) / stride;
                int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
                *pad_head = (pad_needed + 1) / 2;
                *pad_tail = pad_needed - *pad_head;
                *out_size = ComputeOutputSize(in_size, stride, kernel, pad_needed, dilation);
                break;
            }
            case AutoPadType::SAME_UPPER: {
                int64_t legacy_target_size = (in_size + stride - 1) / stride;
                int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
                *pad_head = pad_needed / 2;
                *pad_tail = pad_needed - *pad_head;
                *out_size = ComputeOutputSize(in_size, stride, kernel, pad_needed, dilation);
                break;
            }
            default:
                throw std::runtime_error("ComputeSizePadDilations: unexpected AutoPadType.");
        }
    }
    else {
        *out_size = ComputeOutputSize(in_size, stride, kernel, *pad_head + *pad_tail, dilation);
    }
}
 

template<typename T>
int64_t MaxPool<T>::ComputeOutputSize(int64_t in_size,
                                      int64_t stride,
                                      int64_t kernel,
                                      int64_t pad_needed,
                                      int64_t dilation) const {
    if (ceil_mode_ == 0)
        return static_cast<int64_t>(static_cast<float>(
            in_size + pad_needed - dilation * (kernel - 1) - 1) / stride + 1);
    return static_cast<int64_t>(
        std::ceil(static_cast<float>(
            in_size + pad_needed - dilation * (kernel - 1) - 1) / stride + 1));
}

  
template<typename T>
py::tuple MaxPool<T>::compute(py::array_t<T, py::array::c_style | py::array::forcecast> X) const {

    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);

    if (x_dims.size() < 3)
        throw std::runtime_error("Number of dimensions for input should be >= 3.");
    if (kernel_shape_.size() != x_dims.size() - 2) {
        char buffer[1000];
        sprintf(buffer, "Dimension mismatch between kernel_shape (%d) and input dimensions (%d) - 2.",
                (int)kernel_shape_.size(), (int)x_dims.size());
        throw std::runtime_error(buffer);
    }

    std::vector<int64_t> dilations = dilations_;
    if (dilations.size() == 0)
        dilations.resize(x_dims.size(), (int64_t)1);

    std::vector<int64_t> strides = strides_;
    if (strides.size() == 0)
        strides.resize(x_dims.size(), (int64_t)1);
    
    bool need_dilation = false;
    for (auto n : dilations)
        need_dilation |= n > 1;

    std::vector<int64_t> pads = pads_;
    if (pads.size() == 0)
        pads.resize(kernel_shape_.size() * 2 > (x_dims.size() - 2) * 2 
                        ? kernel_shape_.size() * 2 : (x_dims.size() - 2) * 2,
                    (int64_t)0);
    std::vector<int64_t> kernel_shape = kernel_shape_;
    std::vector<int64_t> output_dims = SetOutputSize(x_dims, x_dims[1], &pads, &strides,
                                                     &kernel_shape, &dilations);

    py::array_t<T> Y(output_dims);
    py::array_t<int64_t> I(output_dims);
    {
        py::gil_scoped_release release;
        compute_gil_free(X, Y, &I, kernel_shape, pads, strides, dilations, x_dims, output_dims);
    }
    return py::make_tuple(Y, I);
}


struct TensorOpCost {
    double bytes_loaded;
    double bytes_stored;
    double compute_cycles;
};


template <typename T>
struct MaxPool1DTask final {
    const T* X_data;
    T* Y_data;
    int64_t* I_data;
    int64_t x_step;
    int64_t y_step;
    int64_t dilation_h;
    int64_t pooled_height;
    int64_t stride_h;
    int64_t height;
    const std::vector<int64_t>& kernel_shape;
    const std::vector<int64_t>& pads;
    
    TensorOpCost Cost() {
        double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
        return TensorOpCost{loop_count, loop_count, loop_count};
    }

    void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t c = begin; c < end; ++c)
            operator()(c);
    }

    void operator()(std::ptrdiff_t c) const {
        const T* x_d = X_data + c * x_step;
        T* y_d = Y_data + c * y_step;
        int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
        for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h - pads[0];
            int64_t hend = hstart + kernel_shape[0] * dilation_h;
            T Yh = std::numeric_limits<T>::lowest();
            int64_t h_index = -1;
            for (int64_t h = hstart; h < hend; h += dilation_h) {
                if ((static_cast<uint64_t>(h) < static_cast<uint64_t>(height)) &&
                        (x_d[h] > Yh)) {
                    Yh = x_d[h];
                    h_index = h;
                }
            }
            y_d[ph] = Yh;
            if (i_d != nullptr)
                i_d[ph] = c * x_step + h_index;
        }
    }
};


template <typename T>
struct MaxPool2DTask final {
    const T* X_data;
    T* Y_data;
    int64_t* I_data;
    int64_t x_step;
    int64_t y_step;
    int64_t dilation_h;
    int64_t dilation_w;
    int64_t pooled_height;
    int64_t pooled_width;
    int64_t stride_h;
    int64_t stride_w;
    int64_t height;
    int64_t width;
    const std::vector<int64_t>& kernel_shape;
    const std::vector<int64_t>& pads;
    int64_t storage_order;

    TensorOpCost Cost() {
        double loop_count = static_cast<double>(
            pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
        return TensorOpCost{loop_count, loop_count, loop_count};
    }

    void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t c = begin; c < end; ++c)
            operator()(c);
    }

    void operator()(std::ptrdiff_t c) const {
        const T* x_d = X_data + c * x_step;
        T* y_d = Y_data + c * y_step;
        int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
        for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h - pads[0];
            int64_t hend = hstart + kernel_shape[0] * dilation_h;
            for (int64_t pw = 0; pw < pooled_width; ++pw) {
                int64_t wstart = pw * stride_w - pads[1];
                int64_t wend = wstart + kernel_shape[1] * dilation_w;
                const int64_t pool_index = ph * pooled_width + pw;
                T Yh = std::numeric_limits<T>::lowest();
                int64_t h_index = -1;
                int64_t w_index = -1;
                for (int64_t h = hstart; h < hend; h += dilation_h) {
                    if (static_cast<uint64_t>(h) < static_cast<uint64_t>(height)) {
                        for (int64_t w = wstart; w < wend; w += dilation_w) {
                            if (static_cast<uint64_t>(w) < static_cast<uint64_t>(width)) {
                                const int64_t input_index = h * width + w;
                                if (x_d[input_index] > Yh) {
                                    Yh = x_d[input_index];
                                    h_index = h;
                                    w_index = w;
                                }
                            }
                        }
                    }
                }
                y_d[pool_index] = Yh;
                if (i_d != nullptr)
                    i_d[pool_index] =
                        storage_order == 0 ? c * x_step + h_index * width + w_index
                                           : c * x_step + h_index + w_index * height;
            }
        }
    }
};


template <typename T>
struct MaxPool3DTask {
    const T* X_data;
    T* Y_data;
    int64_t* I_data;
    int64_t x_step;
    int64_t y_step;
    int64_t dilation_h;
    int64_t dilation_w;
    int64_t dilation_d;
    int64_t pooled_height;
    int64_t pooled_width;
    int64_t pooled_depth;
    int64_t stride_h;
    int64_t stride_w;
    int64_t stride_d;
    int64_t height;
    int64_t width;
    int64_t depth;
    const std::vector<int64_t>& kernel_shape;
    const std::vector<int64_t>& pads;
    int64_t storage_order;

    void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t c = begin; c < end; ++c)
            operator()(c);
    }

    TensorOpCost Cost() {
        double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                                kernel_shape[1] * kernel_shape[2]);
        return TensorOpCost{loop_count, loop_count, loop_count};
    }

    void operator()(std::ptrdiff_t c) const {
        const T* x_d = X_data + c * x_step;
        T* y_d = Y_data + c * y_step;
        int64_t* i_d = I_data ? I_data + c * y_step : nullptr;

        for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h - pads[0];
            int64_t hend = hstart + kernel_shape[0] * dilation_h;
            for (int64_t pw = 0; pw < pooled_width; ++pw) {
                int64_t wstart = pw * stride_w - pads[1];
                int64_t wend = wstart + kernel_shape[1] * dilation_w;
                for (int64_t pd = 0; pd < pooled_depth; ++pd) {
                    int64_t dstart = pd * stride_d - pads[2];
                    int64_t dend = dstart + kernel_shape[2] * dilation_d;
                    const int64_t pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
                    T Yh = std::numeric_limits<T>::lowest();
                    int64_t h_index = -1;
                    int64_t w_index = -1;
                    int64_t d_index = -1;
                    for (int64_t h = hstart; h < hend; h += dilation_h) {
                        if (static_cast<uint64_t>(h) < static_cast<uint64_t>(height)) {
                            for (int64_t w = wstart; w < wend; w += dilation_w) {
                                if (static_cast<uint64_t>(w) < static_cast<uint64_t>(width)) {
                                    for (int64_t d = dstart; d < dend; d += dilation_d) {
                                        if (static_cast<uint64_t>(d) < static_cast<uint64_t>(depth)) {
                                            const int64_t input_index = h * width * depth + w * depth + d;
                                            if (x_d[input_index] > Yh) {
                                                Yh = x_d[input_index];
                                                h_index = h;
                                                w_index = w;
                                                d_index = d;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    y_d[pool_index] = Yh;
                    if (i_d != nullptr)
                        i_d[pool_index] = storage_order == 0
                                ? c * x_step + h_index * width * depth + w_index * depth + d_index
                                : c * x_step + h_index + w_index * height + d_index * height * width;
                }
            }
        }
    }
};


template<typename T>
void MaxPool<T>::compute_gil_free(
            py::array_t<T> X, py::array_t<T>& Y, py::array_t<int64_t>* I,
            const std::vector<int64_t>& kernel_shape,
            const std::vector<int64_t>& pads,
            const std::vector<int64_t>& strides,
            const std::vector<int64_t>& dilations,
            const std::vector<int64_t>& x_dims,
            const std::vector<int64_t>& y_dims) const {

    const T* X_data = X.data(0);
    T* Y_data = (T*)Y.data(0);
    int64_t* I_data = I != nullptr ? (int64_t*)I->data(0) : nullptr;

    // The main loop
    int64_t channels = x_dims[1];
    int64_t height = x_dims[2];
    int64_t width = kernel_shape.size() > 1 ? x_dims[3] : 1;
    int64_t depth = kernel_shape.size() > 2 ? x_dims[4] : 1;
    int64_t pooled_height = y_dims[2];
    int64_t pooled_width = kernel_shape.size() > 1 ? y_dims[3] : 1;
    int64_t pooled_depth = kernel_shape.size() > 2 ? y_dims[4] : 1;
    const int64_t total_channels = x_dims[0] * channels;
    int64_t stride_h = global_pooling_ ? 1 : strides[0];
    int64_t stride_w = global_pooling_ ? 1 : strides[1];
    int64_t stride_d = global_pooling_ ? 1 : strides[2];

    switch (kernel_shape.size()) {
        case 1: {
            int64_t x_step = height;
            int64_t y_step = pooled_height;
            const int64_t dilation_h = dilations[0];

            MaxPool1DTask<T> task {X_data, Y_data, I_data, x_step, y_step,
                                   dilation_h, pooled_height, stride_h,
                                   height, kernel_shape, pads};
            task(0, total_channels);
            break;
        }

        case 2: {
            int64_t x_step = height * width;
            int64_t y_step = pooled_height * pooled_width;
            const int64_t dilation_h = dilations[0];
            const int64_t dilation_w = dilations[1];
            MaxPool2DTask<T> task {X_data, Y_data, I_data, x_step, y_step, dilation_h,
                                   dilation_w, pooled_height, pooled_width, stride_h,
                                   stride_w, height, width, kernel_shape, pads,
                                   storage_order_};
            task(0, total_channels);
            break;
        }
        
        case 3: {
            int64_t x_step = height * width * depth;
            int64_t y_step = pooled_height * pooled_width * pooled_depth;
            const int64_t dilation_h = dilations[0];
            const int64_t dilation_w = dilations[1];
            const int64_t dilation_d = dilations[2];
            MaxPool3DTask<T> task {X_data, Y_data, I_data, x_step, y_step,
                                   dilation_h, dilation_w, dilation_d, pooled_height, pooled_width,
                                   pooled_depth, stride_h, stride_w, stride_d, height,
                                   width, depth, kernel_shape, pads, storage_order_};
            task(0, total_channels);
            break;
        }
        
        default:
            throw std::runtime_error("MaxPool: not implemented error.");
    }
}


class MaxPoolFloat : public MaxPool<float>
{
    public:
        MaxPoolFloat() : MaxPool<float>() {}
};


class MaxPoolDouble : public MaxPool<double>
{
    public:
        MaxPoolDouble() : MaxPool<double>() {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_max_pool_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements MaxPool operator."
    #else
    R"pbdoc(Implements runtime for operator MaxPool. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/pool.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<MaxPoolFloat> clf (m, "MaxPoolFloat",
        R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/pool.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &MaxPoolFloat::init,
            "Initializes the runtime with the ONNX attributes.");
    clf.def("compute", &MaxPoolFloat::compute,
            "Computes the output for operator MaxPool.");

    py::class_<MaxPoolDouble> cld (m, "MaxPoolDouble",
        R"pbdoc(Implements float runtime for operator Conv. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/pool.cc>`_
in :epkg:`onnxruntime`. Supports double only.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &MaxPoolDouble::init,
            "Initializes the runtime with the ONNX attributes.");
    cld.def("compute", &MaxPoolDouble::compute,
            "Computes the output for operator MaxPool.");
}

#endif
