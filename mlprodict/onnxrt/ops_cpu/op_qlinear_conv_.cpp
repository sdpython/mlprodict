// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/qlinearconv.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "op_qlinear_conv_.hpp"
#include "op_qlinear_cpp_tester_.hpp"
#include "op_qlinear_cpp_qgemm_tester_.hpp"
#include <random>
#include <map>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

class RuntimeTesterQLinearConv : public RuntimeTester {
public:
    RuntimeTesterQLinearConv(const char* op_name, int opset = 13) : RuntimeTester(op_name, opset) {}

    virtual void Run(bool expect_success, const char* ignored = NULL) {
        if (op_name_ == "QLinearConv") {
            RunQLinearConv(expect_success);
            return;
        }
        throw std::invalid_argument(MakeString("Not implemented for ',", op_name_, "'."));
    }

    void RunQLinearConv(bool expect_success) {
        auto t1 = inputs_[0].type_;
        auto t2 = inputs_[3].type_;
        switch (t1) {
        case 3:
            switch (t2) {
            case 3:
                RunTypedQLinearConv<uint8_t, uint8_t>(expect_success);
                break;
            case 4:
                RunTypedQLinearConv<uint8_t, int8_t>(expect_success);
                break;
            default:
                throw std::invalid_argument(MakeString("Not Implemented for type ", t2));
            }
            break;
        case 4:
            switch (t2) {
            case 3:
                RunTypedQLinearConv<int8_t, uint8_t>(expect_success);
                break;
            case 4:
                RunTypedQLinearConv<int8_t, int8_t>(expect_success);
                break;
            default:
                throw std::invalid_argument(MakeString("Not Implemented for type ", t2));
            }
            break;
        default:
            throw std::invalid_argument(MakeString("Not Implemented for type ", t1));
        }
    }

    template <typename T1, typename T2>
    void RunTypedQLinearConv(bool expect_success) {

        QLinearConv<T1, T2> op;
        std::string auto_pad = GetAttribute<std::string>("auto_pad", "NOTSET");
        std::vector<int64_t> kernel_shape = GetVectorAttribute<int64_t>("kernel_shape");
        std::vector<int64_t> dilations = GetVectorAttribute<int64_t>("dilations");
        std::vector<int64_t> pads = GetVectorAttribute<int64_t>("pads");
        std::vector<int64_t> strides = GetVectorAttribute<int64_t>("strides");
        int64_t group = GetAttribute<int64_t>("group", 1);
        op.init(auto_pad, dilations, group, kernel_shape, pads, strides);

        std::vector<shaped_array_t<T1>> res;

        if (inputs_.size() < 9 || inputs_[8].GetVectorValue<int32_t>().size() == 0) {
            shaped_array_t<int32_t> B;
            shaped_array_t<T1> r = op.compute(
                inputs_[0].GetArrayValue<T1>(),  // X
                inputs_[1].GetArrayValue<float>()[0],  // x_scale
                inputs_[2].GetArrayValue<T1>()[0],  // x_zero_point
                inputs_[3].GetArrayValue<T2>(),  // w
                inputs_[4].GetArrayValue<float>(),  // w_scale
                inputs_[5].GetArrayValue<T2>()[0],  // w_zero_point
                inputs_[6].GetArrayValue<float>()[0],  // y_scale
                inputs_[7].GetArrayValue<T1>()[0],  // y_zero_point
                B);
            res.push_back(r);
        }
        else {
            res.push_back(op.compute(
                inputs_[0].GetArrayValue<T1>(),  // X
                inputs_[1].GetArrayValue<float>()[0],  // x_scale
                inputs_[2].GetArrayValue<T1>()[0],  // x_zero_point
                inputs_[3].GetArrayValue<T2>(),  // w
                inputs_[4].GetArrayValue<float>(),  // w_scale
                inputs_[5].GetArrayValue<T2>()[0],  // w_zero_point
                inputs_[6].GetArrayValue<float>()[0],  // y_scale
                inputs_[7].GetArrayValue<T1>()[0],  // y_zero_point
                inputs_[8].GetArrayValue<int32_t>()  // b
            ));
        }
        CheckSameType(res);
    }
};


template <typename T>
struct RequantizeValues {
    RequantizeValues(int32_t zero_point) {
        min_value_ = static_cast<float>(static_cast<int32_t>(std::numeric_limits<T>::min()) - zero_point);
        max_value_ = static_cast<float>(static_cast<int32_t>(std::numeric_limits<T>::max()) - zero_point);
        zero_point_ = static_cast<float>(zero_point);
    }
    float min_value_;
    float max_value_;
    float zero_point_;
};


inline float RoundHalfToEven(float input) {
    if (!std::isfinite(input)) {
        return input;
    }
    // std::remainder returns x - n, where n is the integral value nearest to x. When |x - n| = 0.5, n is chosen to be even
    return input - std::remainderf(input, 1.f);
}


template <typename T>
T RequantizeOutput(int32_t sum, float scale, RequantizeValues<T>& requantize_values) {
    float f = static_cast<float>(sum) * scale;
    f = std::min(f, requantize_values.max_value_);
    f = std::max(f, requantize_values.min_value_);
    return static_cast<T>(RoundHalfToEven(f) + requantize_values.zero_point_);
}


template <typename T1, typename T2>
class QLinearConvOpTester {
protected:
    template <typename T>
    struct QuantizedTensor {
        std::vector<T> data_;
        std::vector<int64_t> shape_;
        std::vector<float> scale_;
        T zero_point_{ 0 };
    };

    std::default_random_engine generator_{ 1234 };
    QuantizedTensor<T1> X_;
    QuantizedTensor<T2> W_;
    std::vector<int32_t> B_;
    std::vector<int64_t> pads_;
    std::vector<int64_t> strides_;
    std::vector<int64_t> dilations_;
    int64_t groups_{ 0 };
    float output_scale_{ 1.0f };
    T1 output_zero_point_{ 0 };

    static size_t ShapeSize(const std::vector<int64_t>& shape) {
        return static_cast<size_t>(std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>()));
    }

    template <typename T>
    void GenerateRandom(
        QuantizedTensor<T>& tensor, const std::vector<int64_t>& shape,
        float scale, T zero_point, int32_t min_value, int32_t max_value, bool random) {
        std::uniform_int_distribution<int32_t> distribution(min_value, max_value);
        size_t shape_size = ShapeSize(shape);
        tensor.data_.resize(shape_size);
        for (size_t n = 0; n < shape_size; n++) {
            tensor.data_[n] = static_cast<T>(random
                ? distribution(generator_)
                : (n % (max_value - min_value) + min_value));
        }
        tensor.shape_ = shape;
        tensor.scale_ = { scale };
        tensor.zero_point_ = { zero_point };
    }

    static bool NextPosition(int64_t N, const int64_t* shape, int64_t* dims) {
        // Loop over spatial axes in reverse order to choose an index, like counting.
        bool incremented = false;
        for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
            int64_t d_max = shape[d_i];
            if (dims[d_i] >= d_max)
                throw std::invalid_argument("Unexpected error");
            if (dims[d_i] == d_max - 1) {
                dims[d_i] = 0;
            }
            else {  // dims[d_i] < d_max - 1
                ++dims[d_i];
                incremented = true;
                break;
            }
        }
        return incremented;
    }

    void ComputeExpectedOutput(std::vector<T1>& Y_data, std::vector<int64_t>& Y_shape) {
        StaticComputeExpectedOutput(
            X_, W_, B_, pads_, strides_, dilations_, groups_,
            output_scale_, output_zero_point_, Y_data, Y_shape);
    }

    static void StaticComputeExpectedOutput(
        const QuantizedTensor<T1>& X_, const QuantizedTensor<T2>& W_,
        const std::vector<int32_t>& B_, const std::vector<int64_t>& pads_,
        const std::vector<int64_t>& strides_, const std::vector<int64_t>& dilations_,
        int64_t groups_, float output_scale_, T1 output_zero_point_,
        std::vector<T1>& Y_data, std::vector<int64_t>& Y_shape) {
        if (W_.shape_.size() <= 2)
            throw std::invalid_argument("Unexpected error.");
        if (X_.shape_.size() != W_.shape_.size())
            throw std::invalid_argument("Unexpected error.");

        const size_t kernel_rank = W_.shape_.size() - 2;
        const int64_t batch_count = X_.shape_[0];
        const int64_t input_channels = X_.shape_[1];
        const int64_t output_channels = W_.shape_[0];
        const int64_t group_count = std::max<int64_t>(groups_, 1LL);
        const int64_t group_input_channels = W_.shape_[1];
        const int64_t group_output_channels = output_channels / group_count;

        if (input_channels != group_input_channels * group_count)
            throw std::invalid_argument("Unexpected error.");
        if (output_channels != group_output_channels * group_count)
            throw std::invalid_argument("Unexpected error.");

        const int64_t* input_shape = X_.shape_.data() + 2;
        const int64_t* kernel_shape = W_.shape_.data() + 2;

        std::vector<int64_t> pads(pads_);
        if (pads.empty())
            pads.resize(kernel_rank * 2, 0);
        std::vector<int64_t> dilations(dilations_);
        if (dilations.empty())
            dilations.resize(kernel_rank, 1);
        std::vector<int64_t> strides(strides_);
        if (strides.empty())
            strides.resize(kernel_rank, 1);

        // Compute the expected shape of the output.
        Y_shape.reserve(kernel_rank + 2);
        Y_shape.push_back(batch_count);
        Y_shape.push_back(output_channels);
        for (size_t n = 0; n < kernel_rank; n++) {
            Y_shape.push_back(
                ((input_shape[n] + pads[n] + pads[kernel_rank + n]) -
                    (dilations[n] * (kernel_shape[n] - 1) + 1)) / strides[n] + 1);
        }
        const int64_t* output_shape = Y_shape.data() + 2;
        Y_data.resize(ShapeSize(Y_shape));

        const int64_t input_image_size = std::accumulate(
            input_shape, input_shape + kernel_rank, 1LL, std::multiplies<int64_t>());
        const int64_t kernel_size = std::accumulate(
            kernel_shape, kernel_shape + kernel_rank, 1LL, std::multiplies<int64_t>());
        const int32_t X_zero_point = X_.zero_point_;
        const int32_t W_zero_point = W_.zero_point_;

        const T1* Xdata = X_.data_.data();
        T1* Ydata = Y_data.data();

        RequantizeValues<T1> requantize_values(output_zero_point_);

        for (int64_t batch = 0; batch < batch_count; batch++) {
            const T2* weight_group = W_.data_.data();
            for (int64_t group = 0; group < group_count; group++) {
                const T2* weight_row = weight_group;

                for (int64_t oc = 0; oc < group_output_channels; oc++) {
                    int64_t channel_index = group * group_output_channels + oc;
                    int32_t bias = B_.empty() ? 0 : B_[channel_index];
                    float weight_scale = W_.scale_[(W_.scale_.size() == 1) ? 0 : channel_index];
                    float requantize_scale = (X_.scale_[0] * weight_scale) / output_scale_;

                    std::vector<int64_t> d_output(kernel_rank, 0);
                    std::vector<int64_t> d_kernel(kernel_rank, 0);
                    do {
                        int32_t sum = bias;
                        const T1* input_image = Xdata;
                        const T2* weight_data = weight_row;
                        for (int64_t ic = 0; ic < group_input_channels; ic++) {
                            do {
                                int64_t input_offset = 0;
                                bool is_padding = false;
                                for (size_t axis = 0; axis < kernel_rank; ++axis) {
                                    int64_t input_dim = (
                                        d_kernel[axis] * dilations[axis] +
                                        d_output[axis] * strides[axis] - pads[axis]);
                                    is_padding |= !is_a_ge_zero_and_a_lt_b(input_dim, input_shape[axis]);
                                    input_offset *= input_shape[axis];
                                    input_offset += input_dim;
                                }
                                int32_t w_value = static_cast<int32_t>(*weight_data++) - W_zero_point;
                                if (!is_padding) {
                                    int32_t x_value = static_cast<int32_t>(input_image[input_offset]) -
                                        X_zero_point;
                                    sum += x_value * w_value;
                                }
                            } while (NextPosition(kernel_rank, kernel_shape, d_kernel.data()));

                            input_image += input_image_size;
                        }
                        *Ydata++ = RequantizeOutput<T1>(sum, requantize_scale, requantize_values);

                    } while (NextPosition(kernel_rank, output_shape, d_output.data()));

                    weight_row += group_input_channels * kernel_size;
                }

                Xdata += group_input_channels * input_image_size;
                weight_group += group_output_channels * group_input_channels * kernel_size;
            }
        }
    }

public:
    QLinearConvOpTester() { }

    void GenerateRandomInput(const std::vector<int64_t>& shape, float scale, T1 zero_point, bool random) {
        GenerateRandom(X_, shape, scale, zero_point, 0, 63, random);
    }

    void GenerateRandomWeights(const std::vector<int64_t>& shape, float scale, T2 zero_point, bool random) {
        if (std::is_signed<T2>::value)
            GenerateRandom(W_, shape, scale, zero_point, -63, 63, random);
        else
            GenerateRandom(W_, shape, scale, zero_point, 0, 255, random);
    }

    void SetWeightScales(const std::vector<float>& scales) {
        W_.scale_ = scales;
    }

    void GenerateRandomBias(bool random) {
        if (W_.shape_.size() < 1)
            throw std::invalid_argument("Unexpected error.");
        const size_t output_channels = static_cast<size_t>(W_.shape_[0]);
        B_.resize(output_channels);
        std::uniform_int_distribution<int32_t> distribution(-423, 423);
        for (size_t n = 0; n < output_channels; n++)
            B_[n] = random ? distribution(generator_) : (n % (423 + 423) - 423);
    }

    void SetPads(const std::vector<int64_t>& pads) {
        pads_ = pads;
    }

    void SetStrides(const std::vector<int64_t>& strides) {
        strides_ = strides;
    }

    void SetDilations(const std::vector<int64_t>& dilations) {
        dilations_ = dilations;
    }

    void SetGroups(int64_t groups) {
        groups_ = groups;
    }

    void SetOutputScaleAndZeroPoint(float output_scale, T1 output_zero_point) {
        output_scale_ = output_scale;
        output_zero_point_ = output_zero_point;
    }

    void Run() {
        RuntimeTesterQLinearConv test("QLinearConv", 10);

        std::vector<T1> Y_data;
        std::vector<int64_t> Y_shape;
        ComputeExpectedOutput(Y_data, Y_shape);

        test.AddInput<T1>("x", X_.shape_, X_.data_);
        test.AddInput<float>("x_scale", {}, X_.scale_);
        test.AddInput<T1>("x_zero_point", {}, { X_.zero_point_ });

        const std::vector<int64_t> W_scale_shape{ static_cast<int64_t>(W_.scale_.size()) };
        test.AddInput<T2>("w", W_.shape_, W_.data_);
        test.AddInput<float>("w_scale", W_scale_shape, W_.scale_);
        test.AddInput<T2>("w_zero_point", {}, { W_.zero_point_ });

        test.AddInput<float>("y_scale", {}, { output_scale_ });
        test.AddInput<T1>("y_zero_point", {}, { output_zero_point_ });

        if (!B_.empty()) {
            const std::vector<int64_t> B_shape{ static_cast<int64_t>(B_.size()) };
            test.AddInput<int32_t>("b", B_shape, B_);
        }

        float abs_error = 0.0f;
        test.AddOutput<uint8_t>("y", Y_shape, Y_data, true, abs_error);

        if (!pads_.empty())
            test.AddAttribute("pads", pads_);
        if (!strides_.empty())
            test.AddAttribute("strides", strides_);
        if (!dilations_.empty())
            test.AddAttribute("dilations", dilations_);
        if (groups_ > 0)
            test.AddAttribute("group", groups_);
        test.Run(true, "");
    }
};

///////////////////////////////////////////////////////////////////////////////

void test_qlinear_qgemm_ii() {
    QgemmU8X8Test<int8_t, int32_t> test;
    test.ExecuteShort();
    test.ExecuteLong();
}

void test_qlinear_qgemm_ui() {
    QgemmU8X8Test<uint8_t, int32_t> test;
    test.ExecuteShort();
    test.ExecuteLong();
}

void test_qlinear_qgemm_if() {
    QgemmU8X8Test<int8_t, float> test;
    test.ExecuteShort();
    test.ExecuteLong();
}

void test_qlinear_qgemm_uf() {
    QgemmU8X8Test<uint8_t, float> test;
    test.ExecuteShort();
    test.ExecuteLong();
}

///////////////////////////////////////////////////////////////////////////////

void test_qlinear_conv_Conv1D_U8S8(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 3, 24, 15 }, .05f, 4, random);
    test.GenerateRandomWeights({ 32, 24, 3 }, .125f, 0, random);
    test.GenerateRandomBias(random);
    test.SetPads({ 1, 1 });
    test.SetOutputScaleAndZeroPoint(.55f, 54);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8S8(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 3, 24, 15, 11 }, .05f, 4, random);
    test.GenerateRandomWeights({ 32, 24, 3, 3 }, .125f, 0, random);
    test.GenerateRandomBias(random);
    test.SetPads({ 1, 1, 1, 1 });
    test.SetOutputScaleAndZeroPoint(.55f, 54);
    test.Run();
}

void test_qlinear_conv_Conv3D_U8S8(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 2, 2, 15, 11, 6 }, .05f, 4, random);
    test.GenerateRandomWeights({ 5, 2, 3, 3, 3 }, .125f, 0, random);
    test.GenerateRandomBias(random);
    test.SetPads({ 1, 1, 1, 1, 1, 1 });
    test.SetOutputScaleAndZeroPoint(.55f, 54);
    test.Run();
}

void test_qlinear_conv_Conv1D_U8S8_Pointwise(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 3, 24, 15 }, .05f, 4, random);
    test.GenerateRandomWeights({ 32, 24, 1 }, .125f, 0, random);
    test.GenerateRandomBias(random);
    test.SetOutputScaleAndZeroPoint(.55f, 54);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8S8_Pointwise(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 3, 24, 15, 11 }, .05f, 4, random);
    test.GenerateRandomWeights({ 32, 24, 1, 1 }, .125f, 0, random);
    test.GenerateRandomBias(random);
    test.SetOutputScaleAndZeroPoint(.55f, 54);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8U8_Pointwise(bool random) {
    QLinearConvOpTester<uint8_t, uint8_t> test;
    test.GenerateRandomInput({ 3, 24, 19, 19 }, .05f, 4, random);
    test.GenerateRandomWeights({ 32, 24, 1, 1 }, .105f, 126, random);
    test.GenerateRandomBias(random);
    test.SetOutputScaleAndZeroPoint(.75f, 114);
    test.Run();
}

void test_qlinear_conv_Conv3D_U8S8_Pointwise(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 2, 2, 15, 11, 6 }, .05f, 4, random);
    test.GenerateRandomWeights({ 5, 2, 1, 1, 1 }, .125f, 0, random);
    test.GenerateRandomBias(random);
    test.SetOutputScaleAndZeroPoint(.55f, 54);
    test.Run();
}

void test_qlinear_conv_Conv1D_U8S8_Dilations(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 4, 19 }, .02f, 20, random);
    test.GenerateRandomWeights({ 6, 4, 3 }, .11f, 0, random);
    test.SetDilations({ 2 });
    test.SetOutputScaleAndZeroPoint(.24f, 15);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8S8_Dilations(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 4, 19, 16 }, .02f, 20, random);
    test.GenerateRandomWeights({ 6, 4, 3, 2 }, .11f, 0, random);
    test.SetDilations({ 2, 2 });
    test.SetOutputScaleAndZeroPoint(.24f, 15);
    test.Run();
}

void test_qlinear_conv_Conv3D_U8S8_Dilations(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 2, 19, 16, 8 }, .02f, 20, random);
    test.GenerateRandomWeights({ 6, 2, 3, 2, 2 }, .11f, 0, random);
    test.SetDilations({ 2, 2, 2 });
    test.SetOutputScaleAndZeroPoint(.24f, 15);
    test.Run();
}

void test_qlinear_conv_Conv1D_U8S8_Strides(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 7, 18 }, .04f, 16, random);
    test.GenerateRandomWeights({ 5, 7, 2 }, .14f, 0, random);
    test.SetStrides({ 2 });
    test.SetOutputScaleAndZeroPoint(.31f, 30);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8S8_Strides(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 7, 18, 24 }, .04f, 16, random);
    test.GenerateRandomWeights({ 5, 7, 2, 3 }, .14f, 0, random);
    test.SetStrides({ 2, 2 });
    test.SetOutputScaleAndZeroPoint(.31f, 30);
    test.Run();
}

void test_qlinear_conv_Conv3D_U8S8_Strides(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 3, 18, 24, 18 }, .04f, 16, random);
    test.GenerateRandomWeights({ 2, 3, 2, 3, 2 }, .14f, 0, random);
    test.SetStrides({ 2, 2, 2 });
    test.SetOutputScaleAndZeroPoint(.31f, 30);
    test.Run();
}

void test_qlinear_conv_Conv1D_U8S8_Groups(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 8, 13 }, .03f, 7, random);
    test.GenerateRandomWeights({ 12, 4, 3 }, .10f, 0, random);
    test.GenerateRandomBias(random);
    test.SetPads({ 1, 1 });
    test.SetGroups(2);
    test.SetOutputScaleAndZeroPoint(.76f, 88);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8S8_Groups(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 8, 13, 17 }, .03f, 7, random);
    test.GenerateRandomWeights({ 12, 4, 3, 3 }, .10f, 0, random);
    test.GenerateRandomBias(random);
    test.SetPads({ 1, 1, 1, 1 });
    test.SetGroups(2);
    test.SetOutputScaleAndZeroPoint(.76f, 88);
    test.Run();
}

void test_qlinear_conv_Conv3D_U8S8_Groups(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 2, 4, 13, 17, 13 }, .03f, 7, random);
    test.GenerateRandomWeights({ 6, 2, 3, 3, 3 }, .10f, 0, random);
    test.GenerateRandomBias(random);
    test.SetPads({ 1, 1, 1, 1, 1, 1 });
    test.SetGroups(2);
    test.SetOutputScaleAndZeroPoint(.76f, 88);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8S8_Groups_PerChannel(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 8, 13, 17 }, .03f, 7, random);
    test.GenerateRandomWeights({ 10, 4, 3, 3 }, .10f, 0, random);
    test.SetWeightScales({ .15f, .14f, .11f, .13f, .15f, .09f, .12f, .16f, .17f, .07f });
    test.GenerateRandomBias(random);
    test.SetPads({ 1, 1, 1, 1 });
    test.SetGroups(2);
    test.SetOutputScaleAndZeroPoint(.76f, 88);
    test.Run();
}

void test_qlinear_conv_Conv2D_U8S8_Groups_Pointwise(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 12, 17, 13 }, .03f, 7, random);
    test.GenerateRandomWeights({ 15, 4, 1, 1 }, .10f, 0, random);
    test.GenerateRandomBias(random);
    test.SetGroups(3);
    test.SetOutputScaleAndZeroPoint(.26f, 88);
    test.Run();
}

void test_qlinear_conv_Conv3D_U8S8_Groups_Pointwise(bool random) {
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 2, 4, 13, 17, 13 }, .03f, 7, random);
    test.GenerateRandomWeights({ 6, 2, 1, 1, 1 }, .10f, 0, random);
    test.GenerateRandomBias(random);
    test.SetGroups(2);
    test.SetOutputScaleAndZeroPoint(.26f, 88);
    test.Run();
}

void test_qlinear_conv_Conv1D_U8S8_Depthwise(bool random) {
    for (int64_t channels : std::initializer_list<int64_t>{ 7, 8, 9, 16, 25, 64 }) {
        QLinearConvOpTester<uint8_t, int8_t> test;
        test.GenerateRandomInput({ 1, channels, 25 }, .03f, 12, random);
        test.GenerateRandomWeights({ channels, 1, 3 }, .10f, 2, random);
        test.GenerateRandomBias(random);
        test.SetPads({ 1, 1 });
        test.SetGroups(channels);
        test.SetOutputScaleAndZeroPoint(.21f, 88);
        test.Run();
    }
}

void test_qlinear_conv_Conv2D_U8S8_Depthwise(bool random) {
    for (int64_t channels : std::initializer_list<int64_t>{ 7, 8, 9, 16, 25, 64 }) {
        QLinearConvOpTester<uint8_t, int8_t> test;
        test.GenerateRandomInput({ 1, channels, 25, 25 }, .03f, 12, random);
        test.GenerateRandomWeights({ channels, 1, 5, 5 }, .10f, 0, random);
        test.GenerateRandomBias(random);
        test.SetPads({ 2, 2, 2, 2 });
        test.SetGroups(channels);
        test.SetOutputScaleAndZeroPoint(.76f, 88);
        test.Run();
    }
}

void test_qlinear_conv_Conv2D_U8U8_Depthwise(bool random) {
    for (int64_t channels : std::initializer_list<int64_t>{ 3, 8, 13, 24, 31, 64 }) {
        QLinearConvOpTester<uint8_t, uint8_t> test;
        test.GenerateRandomInput({ 1, channels, 25, 25 }, .03f, 12, random);
        test.GenerateRandomWeights({ channels, 1, 3, 3 }, .10f, 167, random);
        test.GenerateRandomBias(random);
        test.SetPads({ 2, 0, 2, 0 });
        test.SetGroups(channels);
        test.SetOutputScaleAndZeroPoint(.76f, 88);
        test.Run();
    }
}

void test_qlinear_conv_Conv2D_U8S8_DepthwisePointwise(bool random) {
    // Tests the combination of using the depthwise convolution path along with the
    // pointed convolution optimization that avoids im2col.
    QLinearConvOpTester<uint8_t, int8_t> test;
    test.GenerateRandomInput({ 1, 27, 18, 18 }, .03f, 12, random);
    test.GenerateRandomWeights({ 27, 1, 1, 1 }, .05f, 0, random);
    test.GenerateRandomBias(random);
    test.SetGroups(27);
    test.SetOutputScaleAndZeroPoint(.24f, 88);
    test.Run();
}

void test_qlinear_conv_Conv3D_U8S8_Depthwise(bool random) {
    for (int64_t channels : std::initializer_list<int64_t>{ 6, 8, 31, 64 }) {
        QLinearConvOpTester<uint8_t, int8_t> test;
        test.GenerateRandomInput({ 1, channels, 15, 11, 13 }, .02f, 135, random);
        test.GenerateRandomWeights({ channels, 1, 3, 3, 3 }, .09f, 0, random);
        test.GenerateRandomBias(random);
        test.SetGroups(channels);
        test.SetOutputScaleAndZeroPoint(.85f, 112);
        test.Run();
    }
}

void test_qlinear_conv_Conv2D_U8S8_Requantize_NoBias(bool random) {
    for (int64_t channels = 1; channels <= 32; channels++) {
        QLinearConvOpTester<uint8_t, int8_t> test;
        test.GenerateRandomInput({ 1, 8, 5, 5 }, .05f, 4, random);
        test.GenerateRandomWeights({ channels, 8, 3, 3 }, .125f, 0, random);
        test.SetPads({ 1, 1, 1, 1 });
        test.SetOutputScaleAndZeroPoint(.55f, 56);
        test.Run();
    }
}

void test_qlinear_conv_Conv2D_U8S8_Requantize_Bias(bool random) {
    for (int64_t channels = 1; channels <= 32; channels++) {
        QLinearConvOpTester<uint8_t, int8_t> test;
        test.GenerateRandomInput({ 1, 8, 5, 5 }, .05f, 4, random);
        test.GenerateRandomWeights({ channels, 8, 3, 3 }, .125f, 0, random);
        test.GenerateRandomBias(random);
        test.SetPads({ 1, 1, 1, 1 });
        test.SetOutputScaleAndZeroPoint(.55f, 56);
        test.Run();
    }
}

void test_qlinear_conv_Conv2D_U8S8_Requantize_Bias_PerChannel(bool random) {
    std::vector<float> weight_scales;
    for (int64_t channels = 1; channels <= 32; channels++) {
        QLinearConvOpTester<uint8_t, int8_t> test;
        test.GenerateRandomInput({ 1, 8, 5, 5 }, .05f, 4, random);
        test.GenerateRandomWeights({ channels, 8, 3, 3 }, .125f, 0, random);
        weight_scales.push_back(.120f + .002f * static_cast<float>(channels));
        test.SetWeightScales(weight_scales);
        test.GenerateRandomBias(random);
        test.SetPads({ 1, 1, 1, 1 });
        test.SetOutputScaleAndZeroPoint(.55f, 56);
        test.Run();
    }
}

void TestQGemm0() {
    int64_t output_count = 3;
    int64_t group_output_channels = 3;
    int64_t  lda = 3;
    int64_t ldb = 3;
    int64_t M = 3;
    int64_t kernel_dim = 3;
    uint8_t x_zero_point = 7;
    int8_t w_zero_point = 0;
    bool BIsPacked = false;

    std::vector<uint8_t> worker_gemm_input = {
        6, 7, 8, 9, 10, 13, 26, 39, 1, 14, 27, 40, 0, 13, 26, 39, 1, 14, 27, 40, 2, 15,
        28, 41, 1, 14, 27, 40, 2, 15, 28, 41, 3, 16, 29, 42, 2, 15, 28, 41, 3, 16, 29,
        42, 4, 17, 30, 43, 3, 16, 29, 42, 4, 17, 30, 43, 5, 18, 31, 44, 4, 17, 30, 43,
        5, 18, 31, 44, 6, 19, 32, 45, 5, 18, 31, 44, 6, 19, 32, 45, 7, 20, 33, 46, 6,
        19, 32, 45, 7, 20, 33, 46, 8, 21, 34, 47, 7, 20, 33, 46, 8, 21, 34, 47, 9, 22,
        35, 48, 8, 21, 34, 47, 9, 22, 35, 48, 10, 23, 36, 49, 9, 22, 35, 48, 10, 23, 36,
        49, 11, 24, 37, 50, 10, 23, 36, 49, 11, 24, 37, 50, 12, 25, 38, 51, 11, 24, 37,
        50, 12, 25, 38, 51, 7, 7, 7, 7 };
    std::vector<int8_t> ptrB = {
        -63, -51, -39, -27, -15, -3, 9, 21, 33, 45, 57, -57, -60, -48, -36, -24, -12,
        0, 12, 24, 36, 48, 60, -54, -57, -45, -33, -21, -9, 3, 15, 27, 39, 51, -63,
        -51, -54, -42, -30, -18, -6, 6, 18, 30, 42, 54, -60, -48, -62, -50, -38, -26,
        -14, -2, 10, 22, 34, 46, 58, -56, -59, -47, -35, -23, -11, 1, 13, 25, 37, 49,
        61, -53 };

    std::vector<int32_t> worker_gemm_output(79, -1111);
    std::vector<int32_t> worker_gemm_output_ref(79, -1112);

    QGemm<uint8_t, int8_t>(
        false, false,
        static_cast<size_t>(output_count),  // M
        static_cast<size_t>(group_output_channels),  // N
        static_cast<size_t>(kernel_dim), 1,  // K, alpha
        worker_gemm_input.data(), ptrB.data(), 0,  // A, B, beta
        worker_gemm_output.data(),  // C
        lda, ldb, static_cast<size_t>(M),  // lda, ldb, ldc
        x_zero_point, &w_zero_point,  // ZeroPointA, ZeroPointB
        BIsPacked, false);

    if (worker_gemm_output[0] == -1111)
        throw std::invalid_argument("QGemm failed.");
    if (worker_gemm_output[9] != -1111)
        throw std::invalid_argument("QGemm failed.");

    QgemmU8X8Test<int8_t, int32_t> cl;
    cl.Test(output_count, group_output_channels, kernel_dim, 1,
        worker_gemm_input.data(), lda, x_zero_point,
        (const uint8_t*)ptrB.data(), ldb, (const uint8_t*)&w_zero_point,
        worker_gemm_output.data(), worker_gemm_output_ref.data(), M, false);

    if (worker_gemm_output[10] != -1111)
        throw std::invalid_argument("QGemm failed.");
    if (worker_gemm_output_ref[9] != -1112)
        throw std::invalid_argument("QGemm failed.");
}

void TestQGemm1() {
    int64_t output_count = 13;
    int64_t group_output_channels = 6;
    int64_t  lda = 12;
    int64_t ldb = 12;
    int64_t M = 12;
    int64_t kernel_dim = 12;
    uint8_t x_zero_point = 7;
    int8_t w_zero_point = 0;
    bool BIsPacked = false;

    std::vector<uint8_t> worker_gemm_input = {
        7, 7, 7, 7, 0, 13, 26, 39, 1, 14, 27, 40, 0, 13, 26, 39, 1, 14, 27, 40, 2, 15,
        28, 41, 1, 14, 27, 40, 2, 15, 28, 41, 3, 16, 29, 42, 2, 15, 28, 41, 3, 16, 29,
        42, 4, 17, 30, 43, 3, 16, 29, 42, 4, 17, 30, 43, 5, 18, 31, 44, 4, 17, 30, 43,
        5, 18, 31, 44, 6, 19, 32, 45, 5, 18, 31, 44, 6, 19, 32, 45, 7, 20, 33, 46, 6,
        19, 32, 45, 7, 20, 33, 46, 8, 21, 34, 47, 7, 20, 33, 46, 8, 21, 34, 47, 9, 22,
        35, 48, 8, 21, 34, 47, 9, 22, 35, 48, 10, 23, 36, 49, 9, 22, 35, 48, 10, 23, 36,
        49, 11, 24, 37, 50, 10, 23, 36, 49, 11, 24, 37, 50, 12, 25, 38, 51, 11, 24, 37,
        50, 12, 25, 38, 51, 7, 7, 7, 7 };
    for (size_t i = 0; i < 13 * 12; ++i)
        worker_gemm_input.push_back(worker_gemm_input[i]);
    std::vector<int8_t> ptrB = {
        -63, -51, -39, -27, -15, -3, 9, 21, 33, 45, 57, -57, -60, -48, -36, -24, -12,
        0, 12, 24, 36, 48, 60, -54, -57, -45, -33, -21, -9, 3, 15, 27, 39, 51, -63,
        -51, -54, -42, -30, -18, -6, 6, 18, 30, 42, 54, -60, -48, -62, -50, -38, -26,
        -14, -2, 10, 22, 34, 46, 58, -56, -59, -47, -35, -23, -11, 1, 13, 25, 37, 49,
        61, -53 };
    for (size_t i = 0; i < 6 * 12; ++i)
        ptrB.push_back(ptrB[i]);

    std::vector<int32_t> worker_gemm_output(179, -1111);
    std::vector<int32_t> worker_gemm_output_ref(179, -1112);

    QGemm<uint8_t, int8_t>(
        false, false,
        static_cast<size_t>(output_count),  // M
        static_cast<size_t>(group_output_channels),  // N
        static_cast<size_t>(kernel_dim), 1,  // K, alpha
        worker_gemm_input.data(), ptrB.data(), 0,  // A, B, beta
        worker_gemm_output.data(),  // C
        lda, ldb, static_cast<size_t>(M),  // lda, ldb, ldc
        x_zero_point, &w_zero_point,  // ZeroPointA, ZeroPointB
        BIsPacked, false);

    if (worker_gemm_output[0] == -1111)
        throw std::invalid_argument("QGemm failed.");

    QgemmU8X8Test<int8_t, int32_t> cl;
    cl.Test(output_count, group_output_channels, kernel_dim, 1,
        worker_gemm_input.data(), lda, x_zero_point,
        (const uint8_t*)ptrB.data(), ldb, (const uint8_t*)&w_zero_point,
        worker_gemm_output.data(), worker_gemm_output_ref.data(), M, false);

    if (worker_gemm_output[0] == -1111)
        throw std::invalid_argument("QGemm failed.");
    if (worker_gemm_output_ref[0] == -1112)
        throw std::invalid_argument("QGemm failed.");
}


#ifndef SKIP_PYTHON


class QLinearConvInt8 : public QLinearConv<
    int8_t, int8_t, int8_t, int32_t,
    py_array_t<int8_t, py_array_style>,
    py_array_t<int8_t, py_array_style>,
    py_array_t<int8_t, py_array_style>,
    py_array_t<int32_t, py_array_style>,
    py_array_t<float, py_array_style>> {
public:
    QLinearConvInt8() : QLinearConv<
        int8_t, int8_t, int8_t, int32_t,
        py_array_t<int8_t, py_array_style>,
        py_array_t<int8_t, py_array_style>,
        py_array_t<int8_t, py_array_style>,
        py_array_t<int32_t, py_array_style>,
        py_array_t<float, py_array_style>>() {}
};


class QLinearConvUInt8 : public QLinearConv<
    uint8_t, uint8_t, uint8_t, int32_t,
    py_array_t<uint8_t, py_array_style>,
    py_array_t<uint8_t, py_array_style>,
    py_array_t<uint8_t, py_array_style>,
    py_array_t<int32_t, py_array_style>,
    py_array_t<float, py_array_style>> {
public:
    QLinearConvUInt8() : QLinearConv<
        uint8_t, uint8_t, uint8_t, int32_t,
        py_array_t<uint8_t, py_array_style>,
        py_array_t<uint8_t, py_array_style>,
        py_array_t<uint8_t, py_array_style>,
        py_array_t<int32_t, py_array_style>,
        py_array_t<float, py_array_style>> () {}
};

PYBIND11_MODULE(op_qlinear_conv_, m) {
    m.doc() =
#if defined(__APPLE__)
        "Implements Conv operator."
#else
        R"pbdoc(Implements runtime for operator QLinearConv. The code is inspired from
`conv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/qlinearconv.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
#endif
;

    m.def("test_qlinear_qgemm_ii", &test_qlinear_qgemm_ii, R"pbdoc(Unit test for operator QGemm.)pbdoc");
    m.def("test_qlinear_qgemm_ui", &test_qlinear_qgemm_ui, R"pbdoc(Unit test for operator QGemm.)pbdoc");
    m.def("test_qlinear_qgemm_if", &test_qlinear_qgemm_if, R"pbdoc(Unit test for operator QGemm.)pbdoc");
    m.def("test_qlinear_qgemm_uf", &test_qlinear_qgemm_uf, R"pbdoc(Unit test for operator QGemm.)pbdoc");
    m.def("test_qgemm0", &TestQGemm0, R"pbdoc(Unit test for operator QGemm.)pbdoc");
    m.def("test_qgemm1", &TestQGemm1, R"pbdoc(Unit test for operator QGemm.)pbdoc");

    m.def("test_qlinear_conv_Conv1D_U8S8", &test_qlinear_conv_Conv1D_U8S8, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8", &test_qlinear_conv_Conv2D_U8S8, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv3D_U8S8", &test_qlinear_conv_Conv3D_U8S8, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv1D_U8S8_Pointwise", &test_qlinear_conv_Conv1D_U8S8_Pointwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Pointwise", &test_qlinear_conv_Conv2D_U8S8_Pointwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8U8_Pointwise", &test_qlinear_conv_Conv2D_U8U8_Pointwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv3D_U8S8_Pointwise", &test_qlinear_conv_Conv3D_U8S8_Pointwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv1D_U8S8_Dilations", &test_qlinear_conv_Conv1D_U8S8_Dilations, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Dilations", &test_qlinear_conv_Conv2D_U8S8_Dilations, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv3D_U8S8_Dilations", &test_qlinear_conv_Conv3D_U8S8_Dilations, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv1D_U8S8_Strides", &test_qlinear_conv_Conv1D_U8S8_Strides, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Strides", &test_qlinear_conv_Conv2D_U8S8_Strides, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv3D_U8S8_Strides", &test_qlinear_conv_Conv3D_U8S8_Strides, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv1D_U8S8_Depthwise", &test_qlinear_conv_Conv1D_U8S8_Depthwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Depthwise", &test_qlinear_conv_Conv2D_U8S8_Depthwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8U8_Depthwise", &test_qlinear_conv_Conv2D_U8U8_Depthwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_DepthwisePointwise", &test_qlinear_conv_Conv2D_U8S8_DepthwisePointwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv3D_U8S8_Depthwise", &test_qlinear_conv_Conv3D_U8S8_Depthwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Requantize_NoBias", &test_qlinear_conv_Conv2D_U8S8_Requantize_NoBias, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Requantize_Bias", &test_qlinear_conv_Conv2D_U8S8_Requantize_Bias, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Requantize_Bias_PerChannel", &test_qlinear_conv_Conv2D_U8S8_Requantize_Bias_PerChannel, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Groups_Pointwise", &test_qlinear_conv_Conv2D_U8S8_Groups_Pointwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv3D_U8S8_Groups_Pointwise", &test_qlinear_conv_Conv3D_U8S8_Groups_Pointwise, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Groups", &test_qlinear_conv_Conv2D_U8S8_Groups, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv3D_U8S8_Groups", &test_qlinear_conv_Conv3D_U8S8_Groups, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv1D_U8S8_Groups", &test_qlinear_conv_Conv1D_U8S8_Groups, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");
    m.def("test_qlinear_conv_Conv2D_U8S8_Groups_PerChannel", &test_qlinear_conv_Conv2D_U8S8_Groups_PerChannel, R"pbdoc(Unit test for operator QLinearConv.)pbdoc");

    py::class_<QLinearConvUInt8> clf(m, "QLinearConvUInt8",
        R"pbdoc(Implements uint8 runtime for operator QLinearConvUInt8. The code is inspired from
`qlinearconv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/qlinearconv.cc>`_
in :epkg:`onnxruntime`. Supports uint8 only.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &QLinearConvUInt8::init,
        "Initializes the runtime with the ONNX attributes.");
    clf.def("compute", &QLinearConvUInt8::compute,
        "Computes the output for operator QLinearConv.");

    py::class_<QLinearConvInt8> cld(m, "QLinearConvInt8",
        R"pbdoc(Implements int8 runtime for operator QLinearConv. The code is inspired from
`qlinearconv.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/qlinearconv.cc>`_
in :epkg:`onnxruntime`. Supports int8 only.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &QLinearConvInt8::init,
        "Initializes the runtime with the ONNX attributes.");
    cld.def("compute", &QLinearConvInt8::compute,
        "Computes the output for operator QLinearConv.");
}

#endif
