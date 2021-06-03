// debug.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include "op_qlinear_conv_.hpp"


void test_qlinear_conv() {
	QLinearConv<uint8_t> conv;

	const std::string auto_pad = "NOTSET";
	py_array_t<int64_t> dilations;
	py_array_t<int64_t> kernel_shape;
	py_array_t<int64_t> pads;
	py_array_t<int64_t> strides;

	conv.init(auto_pad, dilations, 1, kernel_shape, pads, strides);

	py_array_t<uint8_t> x({
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

	py_array_t<uint8_t> w({ 0 }, { 1, 1, 1, 1 });

	py_array_t<float> w_scale({ (float)0.00172794575 }, {});
	uint8_t w_zero_point = 255;

	float y_scale = (float)0.00162681262;
	uint8_t y_zero_point = 123;

	py_array_t<uint8_t> B;

	py_array_t<uint8_t> output({
			0, 81, 93, 230, 52, 87, 197,
			240, 196, 18, 160, 126, 255, 191,
			199, 13, 102, 34, 87, 243, 89,
			23, 77, 69, 60, 18, 93, 18,
			67, 216, 131, 178, 175, 153, 212,
			128, 25, 234, 172, 214, 215, 121,
			0, 101, 163, 114, 213, 107, 8 },
		{ 1, 1, 7, 7 });
	py_array_t < uint8_t> res = conv.compute(
		x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B);
	if (!res.equal(output))
		throw std::runtime_error("failed");
}


int main() {
	test_qlinear_conv();
}
