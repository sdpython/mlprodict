#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#include "experimental_c_reduce.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>


template <typename NTYPE>
void vector_add_pointer(NTYPE* acc, const NTYPE* x, size_t size);

template <>
void vector_add_pointer(float* acc, const float* x, size_t size);

void experimental_ut_reduce();
