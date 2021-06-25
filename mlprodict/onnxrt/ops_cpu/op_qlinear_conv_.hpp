#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/qlinearconv.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "op_conv_matrices_.hpp"

#define ROUNDING_BIAS_MAGIC                    12582912.f
#define ROUNDING_BIAS_MAGIC_BITS               0x4B400000


template <typename T>
inline bool is_signed() { return true; }

template <>
inline bool is_signed<uint8_t>() { return false; }

inline uint32_t BitsOfFp32(float FloatValue) {
    union {
        uint32_t IntegerValue;
        float FloatValue;
    } u;
    u.FloatValue = FloatValue;
    return u.IntegerValue;
}


/**
* This routine requantizes the intermediate buffer to the output buffer
* optionally adding the supplied bias.
* Parameters:
* * Input - Supplies the input matrix.
* * Output - Supplies the output matrix.
* * Bias - Supplies the optional bias vector to be added to the input buffer
*         before requantization.
* * Buffer - Supplies the output matrix.
* * M - Supplies the number of elements of the bias vector and the number of
*         rows in the output matrix.
* * N - Supplies the number of columns of the output matrix.
* * Scale - Supplies the quantization scale.
* * PerColumnScale - Supplies true if the quantization scale has per-column
*         values, else false if a single quantization scale applies to the
*         entire matrix.
* * ZeroPoint - Supplies the quantization zero point value.
*/
template <typename T>
void RequantizeOutput(
    const int32_t* Input, T* Output, const int32_t* Bias, size_t M, size_t N,
    const float* Scale, bool PerColumnScale, T ZeroPoint) {
    const float PerMatrixScaleValue = PerColumnScale ? 0.0f : *Scale;
    const float MinimumValue = float(0 - ZeroPoint);
    const float MaximumValue = float(255 - ZeroPoint);

    // Step through each row of the output matrix.
    while (M-- > 0) {

        const int32_t* bias = Bias;
        const float* scale = Scale;
        int32_t IntegerValue;
        size_t n = N;

        while (n > 0) {

            IntegerValue = *Input++;
            if (bias != nullptr)
                IntegerValue += *bias++;

            float FloatValue = float(IntegerValue);
            float ScaleValue = PerColumnScale ? *scale++ : PerMatrixScaleValue;

            FloatValue *= ScaleValue;
            FloatValue = std::max(FloatValue, MinimumValue);
            FloatValue = std::min(FloatValue, MaximumValue);

            // Use the fast rounding trick adapted from XNNPACK: bias the floating
            // point value by the first floating point value that has no
            // fractional bits. The add operation performs the "round to nearest
            // even". Extract the mantissa bits from this floating point value to
            // obtain the rounded integer value.

            IntegerValue = int32_t(BitsOfFp32(FloatValue + ROUNDING_BIAS_MAGIC)) - ROUNDING_BIAS_MAGIC_BITS;
            *Output++ = T(IntegerValue + ZeroPoint);
            n -= 1;
        }
    }
}


template <typename T1, typename T2, typename T3 = T1, typename T4 = int32_t,
    typename AT1 = shaped_array_t<T1>, typename AT2 = shaped_array_t<T2>,
    typename AT3 = shaped_array_t<T3>, typename AT4 = shaped_array_t<T4>,
    typename ATfloat = shaped_array_t<float>>
    class QLinearConv : public ConvPoolCommon {

    private:

        std::vector<int64_t> W_shape_;
        T2* packed_W_buffer_;
        size_t packed_W_size_;
        T2* reordered_W_buffer_;
        bool is_W_signed_;
        bool is_W_packed_;
        bool channels_last_;

    public:

        typedef const T1* T1_const_ptr;

        QLinearConv() : ConvPoolCommon() {}

        void init(
            const std::string& auto_pad, std::vector<int64_t> dilations, int64_t group,
            std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
            ConvPoolCommon::initcpp(auto_pad, dilations, group, kernel_shape, pads, strides);
            is_W_signed_ = is_signed<T2>();
            channels_last_ = false;
            reordered_W_buffer_ = nullptr;
            packed_W_buffer_ = nullptr;
            is_W_packed_ = false;
            packed_W_size_ = 0;
        }

        AT3 compute(
            AT1 X,  // 0
            float x_scale,  // 1
            T1 x_zero_point,  // 2
            AT2 W,  // 3
            ATfloat w_scale,  // 4
            T2 w_zero_point,  // 5
            float y_scale, // 6
            T3 y_zero_point,  // 7
            AT4 B) const {  // 8

            std::vector<int64_t> x_dims;
            arrayshape2vector(x_dims, X);
            std::vector<int64_t> w_dims;
            arrayshape2vector(w_dims, W);

            const int64_t N = x_dims[0];
            const int64_t M = w_dims[0];
            // const auto& W_shape = M > 0 ? w_dims : W_shape_;
            // const bool is_W_signed = is_W_signed_;
            // const Tensor* W = is_W_packed_ ? nullptr : context->Input<Tensor>(3);

            std::vector<float> output_scales;
            std::vector<int64_t> W_scale_shape;
            arrayshape2vector(W_scale_shape, w_scale);
            const int64_t W_scale_size = flattened_dimension(W_scale_shape);
            const float* W_scale_data = w_scale.data();
            output_scales.resize(static_cast<size_t>(W_scale_size));
            for (int64_t i = 0; i < W_scale_size; i++)
                output_scales[i] = (x_scale * W_scale_data[i] / y_scale);

            std::vector<int64_t> kernel_shape;
            compute_kernel_shape(w_dims, kernel_shape);
            const size_t kernel_rank = kernel_shape.size();
            std::vector<int64_t> pads(pads_);
            if (pads.empty())
                pads.resize(kernel_rank * 2, 0);
            std::vector<int64_t> dilations(dilations_);
            if (dilations.empty())
                dilations.resize(kernel_rank, 1);
            std::vector<int64_t> strides(strides_);
            if (strides.empty())
                strides.resize(kernel_rank, 1);

            // const int64_t C = x_dims[channels_last_ ? 1 + kernel_rank : 1];
            const size_t spatial_dim_start = channels_last_ ? 1 : 2;
            const size_t spatial_dim_end = spatial_dim_start + kernel_rank;
            std::vector<int64_t> y_dims({ N });
            if (!channels_last_)
                y_dims.push_back(M);

            std::vector<int64_t> sliced_input_shape(spatial_dim_end - spatial_dim_start);
            for (size_t i = 0; i < sliced_input_shape.size(); ++i)
                sliced_input_shape[i] = x_dims[i + spatial_dim_start];

            infer_output_shape(sliced_input_shape, kernel_shape, strides, dilations, pads, y_dims, false);
            if (channels_last_)
                y_dims.push_back(M);

            std::vector<int64_t> output_shape(y_dims.begin() + spatial_dim_start, y_dims.begin() + spatial_dim_end);
            AT3 Y(y_dims);
            {
#ifndef SKIP_PYTHON
                py_gil_scoped_release release;
#endif
                compute_gil_free(X, W, B, Y,
                    sliced_input_shape, output_shape,
                    kernel_shape, pads, dilations, strides,
                    x_dims, y_dims, w_dims,
                    x_scale, x_zero_point,
                    w_scale, w_zero_point,
                    y_scale, y_zero_point,
                    output_scales);
            }
            return Y;
        }

    private:

        bool HasStridesOneAndNoPadding() const {
            if (std::all_of(strides_.begin(), strides_.end(), [](int64_t v) { return v == 1; })) {
                if (std::all_of(pads_.begin(), pads_.end(), [](int64_t v) { return v == 0; }))
                    return true;
            }
            return false;
        }

        template <typename T>
        void ReorderFilter(
            const T* input, T* output,
            size_t output_channels,
            size_t input_channels,
            size_t kernel_size) const {
            for (size_t k = 0; k < kernel_size; k++) {
                for (size_t ic = 0; ic < input_channels; ic++) {
                    for (size_t oc = 0; oc < output_channels; oc++) {
                        size_t index = (oc * input_channels * kernel_size) + (ic * kernel_size) + k;
                        *output++ = input[index];
                    }
                }
            }
        }

        void compute_gil_free(
            AT1 X, AT2 W, AT4 B, AT3& Y,
            const std::vector<int64_t>& input_shape,
            const std::vector<int64_t>& output_shape,
            const std::vector<int64_t>& kernel_shape,
            const std::vector<int64_t>& pads,
            const std::vector<int64_t>& dilations,
            const std::vector<int64_t>& strides,
            const std::vector<int64_t>& x_dims,
            const std::vector<int64_t>& y_dims,
            const std::vector<int64_t>& w_dims,
            float x_scale, T1 x_zero_point,
            ATfloat w_scale, T2 w_zero_point,
            float y_scale, T3 y_zero_point,
            const std::vector<float>& output_scales) const {

            // see https://github.com/microsoft/onnxruntime/pull/7885/
            std::vector<int64_t> b_dims;
            arrayshape2vector(b_dims, B);

            const int64_t N = x_dims[0];
            const int64_t C = x_dims[1];
            const int64_t M = w_dims[0];

            const int64_t input_image_size = flattened_dimension(input_shape);
            const int64_t output_image_size = flattened_dimension(output_shape);
            const int64_t kernel_size = flattened_dimension(kernel_shape);

            // Handle the case of a dynamic weight filter.
            T2* reordered_W_buffer = nullptr;
            T2* reordered_W = nullptr;
            if (!packed_W_buffer_) {
                if (w_dims.empty()) {
                    // Weight was constant and reordered.
                    reordered_W = reordered_W_buffer_;
                }
                else {
                    // Weight tensor was not constant or prepacking is disabled.
                    reordered_W = new T2[flattened_dimension(w_dims)];
                    reordered_W_buffer = reordered_W;
                    ReorderFilter(
                        W.data(),
                        reordered_W,
                        static_cast<size_t>(M),
                        static_cast<size_t>(w_dims[1]),
                        static_cast<size_t>(kernel_size));
                }
            }

            int64_t group_count = group_;
            int64_t group_input_channels = w_dims[1];
            int64_t group_output_channels = M / group_count;

            // Test for depthwise convolution.
            const bool is_depthwise_conv = (reordered_W != nullptr && group_input_channels == 1 && group_output_channels == 1);
            if (is_depthwise_conv) {
                // Update the input and output channels to the number of groups in order to
                // reuse as much of the below standard convolution path.
                group_input_channels = group_count;
                group_output_channels = group_count;
                group_count = 1;
            }

            const int64_t X_offset = C * input_image_size;
            const int64_t Y_offset = M * output_image_size;
            const int64_t kernel_dim = group_input_channels * kernel_size;
            const int64_t col_buffer_size = kernel_dim * output_image_size;

            // Use an intermediate int32_t buffer for the GEMM computation before
            // requantizing to the output type.
            int32_t* gemm_output_data = new int32_t[Y_offset];
            // int32_t* gemm_output_buffer = gemm_output_data;
            int32_t* gemm_output = gemm_output_data;

            const T1* Xdata = X.data();
            const int32_t* Bdata = (int32_t*)(b_dims.size() > 0 ? B.data() : nullptr);
            T3* Ydata = (T3*)Y.data();

            T1* transpose_input_buffer = nullptr;
            T3* transpose_output_buffer = nullptr;
            T1* transpose_input = nullptr;
            T3* transpose_output = nullptr;
            if (!channels_last_) {
                transpose_input = new T1[X_offset];
                transpose_input_buffer = transpose_input;
                transpose_output = new T3[Y_offset];
                transpose_output_buffer = transpose_output;
            }

            T1_const_ptr* col_buffer = nullptr;
            T1_const_ptr* col_data = nullptr;
            std::vector<T3> padding_data;
            const size_t kernel_rank = kernel_shape.size();

            if (is_depthwise_conv) {
                // Allocate indirection buffer pointers and prepare a padding vector for
                // the im2col transform.
                col_data = new T1_const_ptr[kernel_size * output_image_size];
                col_buffer = col_data;
                padding_data.resize(static_cast<size_t>(C), x_zero_point);
            }
            else if (kernel_size != 1 || !HasStridesOneAndNoPadding()) {
                // Pointwise convolutions can use the original input tensor in place,
                // otherwise a temporary buffer is required for the im2col transform.
                int64_t group_col_buffer_size = (kernel_rank > 2) ? group_count * col_buffer_size : col_buffer_size;
                col_data = new T1_const_ptr[group_col_buffer_size];
                col_buffer = col_data;
            }

            // See onnxruntime.
            int32_t maximum_thread_count = 16;
            constexpr double thread_complexity = static_cast<double>(64 * 1024);
            const double complexity = static_cast<double>(output_image_size) *
                static_cast<double>(group_output_channels) *
                static_cast<double>(kernel_dim);

            // OMP
            #if USE_OPENMP
            int32_t thread_count = maximum_thread_count;
            if (complexity < thread_complexity * maximum_thread_count)
                thread_count = static_cast<int32_t>(complexity / thread_complexity) + 1;
            // Ensure that every thread produces at least one output.
            if (thread_count > output_image_size)
                thread_count = static_cast<int32_t>(output_image_size);
            thread_count = std::min(thread_count, ::omp_get_max_threads());
            #else
            int32_t thread_count = 1;
            #endif

            for (int64_t image_id = 0; image_id < N; ++image_id) {
                const T1* input_data = Xdata;
                T3* output_data = Ydata;

                if (!channels_last_) {
                    // Transpose the input from channels first (NCHW) to channels last (NHWC).
                    TensorTranspose(
                        Xdata,
                        transpose_input_buffer,
                        static_cast<size_t>(C),
                        static_cast<size_t>(input_image_size));
                    input_data = transpose_input_buffer;
                    output_data = transpose_output_buffer;
                }

                // Threaded implementation of ND convolution is not yet supported, so
                // prepare all im2col transformations here.
                if (!is_depthwise_conv && col_buffer && kernel_rank > 2) {
                    for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                        Im2col_NCHW<T1>(
                            input_data + group_id * group_input_channels,
                            group_input_channels,
                            C,
                            input_shape.data(),
                            output_shape.data(),
                            kernel_shape.data(),
                            strides.data(),
                            dilations.data(),
                            pads.data(),
                            static_cast<int64_t>(kernel_rank),
                            ((T1*)col_buffer) + group_id * col_buffer_size,  // sizeof(T*) > sizeof(T)
                            x_zero_point);
                    }
                }

                #if USE_OPENMP
                #pragma omp parallel for
                #endif
                for (int32_t batch_idx = 0; batch_idx < thread_count; ++batch_idx) {
                    int64_t output_start, output_end;
                    std::ptrdiff_t work_per_batch = output_image_size / thread_count;
                    std::ptrdiff_t work_per_batch_extra = output_image_size % thread_count;
                    if (batch_idx < work_per_batch_extra) {
                        output_start = (work_per_batch + 1) * batch_idx;
                        output_end = output_start + work_per_batch + 1;
                    }
                    else {
                        output_start = work_per_batch * batch_idx + work_per_batch_extra;
                        output_end = output_start + work_per_batch;
                    }
                    int64_t output_count = output_end - output_start;

                    int32_t* worker_gemm_output = gemm_output + output_start * M;
                    T3* worker_requantize_output = output_data + output_start * M;

                    if (is_depthwise_conv) {
                        T1_const_ptr* worker_col_buffer = col_buffer + output_start * kernel_size;
                        Im2col_NHWC<T1>(
                            input_data,
                            C,
                            input_shape.data(),
                            output_shape.data(),
                            kernel_shape.data(),
                            strides.data(),
                            dilations.data(),
                            pads.data(),
                            static_cast<ptrdiff_t>(kernel_rank),
                            output_start,
                            output_count,
                            worker_col_buffer,
                            padding_data.data());
                        QConvDepthwise<T1, T2, int32_t>(
                            worker_col_buffer,
                            x_zero_point,
                            reordered_W,
                            w_zero_point,
                            is_signed<T1>(),
                            worker_gemm_output,
                            static_cast<size_t>(M),
                            static_cast<size_t>(output_count),
                            static_cast<size_t>(kernel_size));
                    }
                    else {
                        size_t lda;
                        for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                            // Prepare the im2col transformation or use the input buffer directly for
                            // pointwise convolutions.
                            const T1* worker_gemm_input;
                            const auto* group_input_data = input_data + group_id * group_input_channels;
                            if (col_buffer) {
                                T1* worker_col_buffer = ((T1*)col_buffer) + output_start * kernel_dim;
                                if (kernel_rank == 2) {
                                    Im2col_NHWC<T1>(
                                        group_input_data,
                                        group_input_channels,
                                        C,
                                        input_shape[0],
                                        input_shape[1],
                                        kernel_shape[0],
                                        kernel_shape[1],
                                        dilations[0],
                                        dilations[1],
                                        pads[0],
                                        pads[1],
                                        strides[0],
                                        strides[1],
                                        output_shape[1],
                                        output_start,
                                        output_count,
                                        worker_col_buffer,
                                        x_zero_point);
                                }
                                else if (kernel_rank == 1) {
                                    Im2col_NHWC<T1>(
                                        group_input_data,
                                        group_input_channels,
                                        C,
                                        1,
                                        input_shape[0],
                                        1,
                                        kernel_shape[0],
                                        1,
                                        dilations[0],
                                        0,
                                        pads[0],
                                        1,
                                        strides[0],
                                        output_shape[0],
                                        output_start,
                                        output_count,
                                        worker_col_buffer,
                                        x_zero_point);
                                }
                                else {
                                    // Use the im2col buffer prepared outside the thread, indexed by group.
                                    worker_col_buffer += group_id * col_buffer_size;
                                }
                                worker_gemm_input = worker_col_buffer;
                                lda = kernel_dim;
                            }
                            else {
                                worker_gemm_input = group_input_data + output_start * C;
                                lda = C;
                            }

                            size_t ldb = 0;
                            const T2* ptrB;
                            bool BIsPacked = false;
                            if (packed_W_buffer_) {
                                ptrB = static_cast<const T2*>(packed_W_buffer_) + group_id * packed_W_size_;
                                BIsPacked = true;
                            }
                            else {
                                ptrB = reordered_W + group_id * group_output_channels;
                                ldb = static_cast<size_t>(M);
                            }

                            QGemm<T1, T2>(
                                false, false,
                                static_cast<size_t>(output_count),  // M
                                static_cast<size_t>(group_output_channels),  // N
                                static_cast<size_t>(kernel_dim), 1,  // K, alpha
                                worker_gemm_input, ptrB, 0,  // A, B, beta
                                worker_gemm_output + group_id * group_output_channels,  // C
                                lda, ldb, static_cast<size_t>(M),  // lda, ldb, ldc
                                x_zero_point, &w_zero_point,  // ZeroPointA, ZeroPointB
                                BIsPacked, false);  // BIsPacked, PerColumnZeroPoints
                        }
                    }

                    RequantizeOutput(
                        worker_gemm_output,
                        worker_requantize_output,
                        Bdata,
                        static_cast<size_t>(output_count),
                        static_cast<size_t>(M),
                        output_scales.data(),
                        output_scales.size() > 1,
                        y_zero_point);
                }

                if (!channels_last_) {
                    // Transpose the output from channels last (NHWC) to channels first (NCHW).
                    TensorTranspose(
                        output_data,
                        Ydata,
                        static_cast<size_t>(output_image_size),
                        static_cast<size_t>(M));
                }

                Xdata += X_offset;
                Ydata += Y_offset;
            }

            if (transpose_input != nullptr)
                delete[] transpose_input;
            if (transpose_output != nullptr)
                delete[] transpose_output;
            if (reordered_W_buffer != nullptr)
                delete[] reordered_W_buffer;
            if (col_data != nullptr)
                delete[] col_data;
        }
};

// unit tests

void test_qlinear_conv_Conv1D_U8S8(bool random);
void test_qlinear_conv_Conv2D_U8S8(bool random);
void test_qlinear_conv_Conv3D_U8S8(bool random);
void test_qlinear_conv_Conv1D_U8S8_Pointwise(bool random);
void test_qlinear_conv_Conv2D_U8S8_Pointwise(bool random);
void test_qlinear_conv_Conv2D_U8U8_Pointwise(bool random);
void test_qlinear_conv_Conv3D_U8S8_Pointwise(bool random);
void test_qlinear_conv_Conv1D_U8S8_Dilations(bool random);
void test_qlinear_conv_Conv2D_U8S8_Dilations(bool random);
void test_qlinear_conv_Conv3D_U8S8_Dilations(bool random);
void test_qlinear_conv_Conv1D_U8S8_Strides(bool random);
void test_qlinear_conv_Conv2D_U8S8_Strides(bool random);
void test_qlinear_conv_Conv3D_U8S8_Strides(bool random);
void test_qlinear_conv_Conv1D_U8S8_Groups(bool random);
void test_qlinear_conv_Conv2D_U8S8_Groups(bool random);
void test_qlinear_conv_Conv3D_U8S8_Groups(bool random);
void test_qlinear_conv_Conv2D_U8S8_Groups_PerChannel(bool random);
void test_qlinear_conv_Conv2D_U8S8_Groups_Pointwise(bool random);
void test_qlinear_conv_Conv3D_U8S8_Groups_Pointwise(bool random);
void test_qlinear_conv_Conv1D_U8S8_Depthwise(bool random);
void test_qlinear_conv_Conv2D_U8S8_Depthwise(bool random);
void test_qlinear_conv_Conv2D_U8U8_Depthwise(bool random);
void test_qlinear_conv_Conv2D_U8S8_DepthwisePointwise(bool random);
void test_qlinear_conv_Conv3D_U8S8_Depthwise(bool random);
void test_qlinear_conv_Conv2D_U8S8_Requantize_NoBias(bool random);
void test_qlinear_conv_Conv2D_U8S8_Requantize_Bias(bool random);
void test_qlinear_conv_Conv2D_U8S8_Requantize_Bias_PerChannel(bool random);

void test_qlinear_qgemm_ii();
void test_qlinear_qgemm_ui();
void test_qlinear_qgemm_if();
void test_qlinear_qgemm_uf();
