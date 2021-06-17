#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#include "experimental_c_helper.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>


template <typename TYPE>
void _check_eq(const std::string& eq, const TYPE& sh);

void _split(const std::string& eq, const mapshape_type& sh, mapshape_type& dx);

void _split(const std::string& eq, const std::vector<int64_t>& sh, mapshape_type& dx);

void _equation_split(const std::string& equation, std::string& eqx, std::string& eqy, std::string& eqr);

void _interpret(
    const mapshape_type& dx, const mapshape_type& dy, const std::string& eqr,
    mapshape_type& shape, std::vector<std::pair<char, char>>& c_uni,
    std::vector<char>& c_trp, std::vector<char>& c_sum);

void _inc(const mapshape_type& d, mapshape_type& res);

int64_t prod(const mapshape_type& seq);

int64_t get_index(const std::vector<int64_t>& incs, const std::vector<int64_t>& index);

void get_incs(const mapshape_type& cd, const mapshape_type& shape, std::vector<int64_t>& incs);

void mapshape2shape(const mapshape_type& shape, std::vector<int64_t>& out_shape);

void mapshape2shape(const mapshape_type& shape, std::vector<size_t>& out_shape);

template <typename NTYPE>
NTYPE vector_dot_product_pointer16(const NTYPE* p1, const NTYPE* p2, size_t size);

std::string code_optimisation();

template <>
float vector_dot_product_pointer16(const float* p1, const float* p2, size_t size);

template <typename NTYPE>
NTYPE vector_dot_product_pointer_stride(
    const NTYPE* xp, const NTYPE* yp, size_t size,
    int64_t inc_left, int64_t inc_right);

void set_index(int64_t begin, const std::vector<int64_t>& shape_dims, std::vector<int64_t>& index);

template <typename NTYPE>
void custom_einsum_matmul(
    const NTYPE* x_data, const NTYPE* y_data,
    int64_t loop_size,
    const mapshape_type& cdx, const mapshape_type& cdy, const mapshape_type& shape,
    const std::vector<int64_t>& left_incs, const std::vector<int64_t>& right_incs,
    NTYPE* z_data, int64_t begin, int64_t end, char col_sum);

void experimental_ut_einsum();
