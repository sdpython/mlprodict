// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "op_qlinear_cpp_qgemm_tester_.hpp"


void TestLocalGemm(
	bool TransA,
	bool TransB,
	size_t M,
	size_t N,
	size_t K,
	float alpha,
	const float* A,
	size_t lda,
	const float* B,
	size_t ldb,
	float beta,
	float* C,
	size_t ldc) {
	throw std::exception("Not implemented error.");
}
