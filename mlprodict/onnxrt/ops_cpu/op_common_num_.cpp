
#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "op_common_num_.hpp"


#include <xmmintrin.h>
#include <emmintrin.h>


float vector_dot_product_pointer16_sse(const float *p1, const float *p2)
{
    __m128 c1 = _mm_load_ps(p1);
    __m128 c2 = _mm_load_ps(p2);
    __m128 r1 = _mm_mul_ps(c1, c2);
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm_load_ps(p1);
    c2 = _mm_load_ps(p2);
    r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm_load_ps(p1);
    c2 = _mm_load_ps(p2);
    r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));
    
    p1 += 4;
    p2 += 4;
    
    c1 = _mm_load_ps(p1);
    c2 = _mm_load_ps(p2);
    r1 = _mm_add_ps(r1, _mm_mul_ps(c1, c2));

    float r[4];
    _mm_store_ps(r, r1);

    return r[0] + r[1] + r[2] + r[3];
}

#define BYNF 16

float vector_dot_product_pointer16_sse(const float *p1, const float *p2, size_t size)
{
    float sum = 0;
    size_t i = 0;
    if (size >= BYNF) {
        size_t size_ = size - BYNF;
        for(; i < size_; i += BYNF, p1 += BYNF, p2 += BYNF)
            sum += vector_dot_product_pointer16_sse(p1, p2);
    }
    for(; i < size; ++p1, ++p2, ++i)
        sum += *p1 * *p2;
    return sum;
}


double vector_dot_product_pointer16_sse(const double *p1, const double *p2)
{
    __m128d c1 = _mm_load_pd(p1);
    __m128d c2 = _mm_load_pd(p2);
    __m128d r1 = _mm_mul_pd(c1, c2);
        
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));
    
    p1 += 2;
    p2 += 2;
    
    c1 = _mm_load_pd(p1);
    c2 = _mm_load_pd(p2);
    r1 = _mm_add_pd(r1, _mm_mul_pd(c1, c2));

    double r[4];
    _mm_store_pd(r, r1);

    return r[0] + r[1] + r[2] + r[3];
}

#define BYND 8

double vector_dot_product_pointer16_sse(const double *p1, const double *p2, size_t size)
{
    double sum = 0;
    size_t i = 0;
    if (size >= BYND) {
        size_t size_ = size - BYND;
        for(; i < size_; i += BYND, p1 += BYND, p2 += BYND)
            sum += vector_dot_product_pointer16_sse(p1, p2);
    }
    for(; i < size; ++p1, ++p2, ++i)
        sum += *p1 * *p2;
    return sum;
}

template <>
float vector_dot_product_pointer_sse(const float *p1, const float *p2, size_t size)
{
    return vector_dot_product_pointer16_sse(p1, p2, size);
}

template <>
double vector_dot_product_pointer_sse(const double *p1, const double *p2, size_t size)
{
    return vector_dot_product_pointer16_sse(p1, p2, size);
}

