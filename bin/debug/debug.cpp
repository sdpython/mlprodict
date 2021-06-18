// debug.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include "op_conv_matrices_.hpp"
#include "op_qlinear_conv_.hpp"
#include "op_qlinear_cpp_qgemm_tester_.hpp"
#include "op_qlinear_cpp_tester_.hpp"
#include "experimental_c.h"


void test_qlinear_conv1() {
    QLinearConv<uint8_t, float> conv_mistyped;
    QLinearConv<uint8_t, uint8_t, uint8_t> conv;

    const std::string auto_pad = "NOTSET";
    std::vector<int64_t> dilations;
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;

    conv.init(auto_pad, dilations, 1, kernel_shape, pads, strides);

    shaped_array_t<uint8_t> x({
        255, 174, 162, 25, 203, 168, 58,
        15, 59, 237, 95, 129, 0, 64,
        56, 242, 153, 221, 168, 12, 166,
        232, 178, 186, 195, 237, 162, 237,
        188, 39, 124, 77, 80, 102, 43,
        127, 230, 21, 83, 41, 40, 134,
        255, 154, 92, 141, 42, 148, 247 },
        { 1, 1, 7, 7 });

    float x_scale = (float)0.00369204697;
    uint8_t x_zero_point = 132;

    shaped_array_t<uint8_t> w({ 0 }, { 1, 1, 1, 1 });

    shaped_array_t<float> w_scale({ (float)0.00172794575 }, {});
    uint8_t w_zero_point = 255;

    float y_scale = (float)0.00162681262;
    uint8_t y_zero_point = 123;

    shaped_array_t<int32_t> B;

    shaped_array_t<uint8_t> output({
            0, 81, 93, 230, 52, 87, 197,
            240, 196, 18, 160, 126, 255, 191,
            199, 13, 102, 34, 87, 243, 89,
            23, 77, 69, 60, 18, 93, 18,
            67, 216, 131, 178, 175, 153, 212,
            128, 25, 234, 172, 214, 215, 121,
            0, 101, 163, 114, 213, 107, 8 },
        { 1, 1, 7, 7 });
    shaped_array_t<uint8_t> res = conv.compute(
        x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B);
    if (!res.equal(output))
        throw std::runtime_error("failed");
}

void test_qlinear_qgemm() {
    TestQGemm0();
    TestQGemm1();
    test_qlinear_qgemm_ii();
    test_qlinear_qgemm_ui();
    test_qlinear_qgemm_if();
    test_qlinear_qgemm_uf();
}

void test_qlinear_conv2(bool random) {
    test_qlinear_conv_Conv1D_U8S8_Groups(random);
    test_qlinear_conv_Conv1D_U8S8(random);
    test_qlinear_conv_Conv1D_U8S8_Groups(random);

    test_qlinear_conv_Conv2D_U8S8(random);
    test_qlinear_conv_Conv3D_U8S8(random);
    test_qlinear_conv_Conv1D_U8S8_Pointwise(random);
    test_qlinear_conv_Conv2D_U8S8_Pointwise(random);
    test_qlinear_conv_Conv2D_U8U8_Pointwise(random);
    test_qlinear_conv_Conv3D_U8S8_Pointwise(random);
    test_qlinear_conv_Conv1D_U8S8_Dilations(random);
    test_qlinear_conv_Conv2D_U8S8_Dilations(random);
    test_qlinear_conv_Conv3D_U8S8_Dilations(random);
    test_qlinear_conv_Conv1D_U8S8_Strides(random);
    test_qlinear_conv_Conv2D_U8S8_Strides(random);
    test_qlinear_conv_Conv3D_U8S8_Strides(random);
    test_qlinear_conv_Conv1D_U8S8_Depthwise(random);
    test_qlinear_conv_Conv2D_U8S8_Depthwise(random);
    test_qlinear_conv_Conv2D_U8U8_Depthwise(random);
    test_qlinear_conv_Conv2D_U8S8_DepthwisePointwise(random);
    test_qlinear_conv_Conv3D_U8S8_Depthwise(random);
    test_qlinear_conv_Conv2D_U8S8_Requantize_NoBias(random);
    test_qlinear_conv_Conv2D_U8S8_Requantize_Bias(random);
    test_qlinear_conv_Conv2D_U8S8_Requantize_Bias_PerChannel(random);
    test_qlinear_conv_Conv2D_U8S8_Groups_Pointwise(random);
    test_qlinear_conv_Conv3D_U8S8_Groups_Pointwise(random);
    test_qlinear_conv_Conv2D_U8S8_Groups(random);
    test_qlinear_conv_Conv3D_U8S8_Groups(random);
    test_qlinear_conv_Conv1D_U8S8_Groups(random);
    test_qlinear_conv_Conv2D_U8S8_Groups_PerChannel(random);
}

int main() {
    experimental_ut_add();
    experimental_ut_einsum();
    experimental_ut_reduce();
    TestQGemm0();
    TestQGemm1();
    test_qlinear_conv2(false);
    test_qlinear_conv2(true);
    test_qlinear_conv1();
    test_qlinear_qgemm();
}
