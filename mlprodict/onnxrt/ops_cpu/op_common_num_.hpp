#pragma once

#include <cmath>
#include <vector>
#include <stdio.h>


float vector_dot_product_pointer16_sse(const float *p1, const float *p2, size_t size);

double vector_dot_product_pointer16_sse(const double *p1, const double *p2, size_t size);

template <typename NTYPE>
NTYPE vector_dot_product_pointer_sse(const NTYPE *p1, const NTYPE *p2, size_t size);
