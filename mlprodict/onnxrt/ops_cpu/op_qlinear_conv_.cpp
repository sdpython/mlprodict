// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif


#include "op_qlinear_conv_.hpp"


#ifndef SKIP_PYTHON


class QLinearConvInt8 : public QLinearConv<int8_t> {
public:
	QLinearConvInt8() : QLinearConv<int8_t>() {}
};


class QLinearConvUInt8 : public QLinearConv<uint8_t> {
public:
	QLinearConvUInt8() : QLinearConv<uint8_t>() {}
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
