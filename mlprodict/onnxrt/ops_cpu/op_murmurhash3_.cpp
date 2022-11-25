// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

//scikit-learn is a Python module for machine learning built on top of SciPy and
//distributed under the 3-Clause BSD license. See https://github.com/scikit-learn/scikit-learn.
//This material is licensed under the BSD License (see https://github.com/scikit-learn/scikit-learn/blob/master/COPYING);

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/contrib_ops/cpu//murmur_hash3.cc.


#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef SKIP_PYTHON
//#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include <numpy/arrayobject.h>

#if USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
#endif

#include "op_common_.hpp"


#if defined(_MSC_VER)

#define FORCE_INLINE __forceinline
#include <stdlib.h>
#define ROTL32(x, y) _rotl(x, y)
#define BIG_CONSTANT(x) (x)

#else

    #if defined(GNUC) && ((GNUC > 4) || (GNUC == 4 && GNUC_MINOR >= 4))

    // gcc version >= 4.4 4.1 = RHEL 5, 4.4 = RHEL 6.
    // Don't inline for RHEL 5 gcc which is 4.1
    #define FORCE_INLINE attribute((always_inline))

    #else

    #define FORCE_INLINE

    #endif

inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

#define ROTL32(x, y) rotl32(x, y)
#define BIG_CONSTANT(x) (x##LLU)

#endif


FORCE_INLINE uint32_t getblock(const uint32_t* p, int i) {
    return p[i];
}


FORCE_INLINE uint32_t fmix(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}


uint32_t MurmurHash3_x86_32_void(const void* key, int len, uint32_t seed) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(key);
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;

    const uint32_t* blocks = reinterpret_cast<const uint32_t*>(data + static_cast<int64_t>(nblocks) * 4);

    for (int i = -nblocks; i; i++) {
        uint32_t k1 = getblock(blocks, i);

        k1 *= c1;
        k1 = ROTL32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = ROTL32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t* tail = reinterpret_cast<const uint8_t*>(data + static_cast<int64_t>(nblocks) * 4);

    uint32_t k1 = 0;

    switch (len & 3) {
        case 3:
            k1 ^= tail[2] << 16;
        case 2:
            k1 ^= tail[1] << 8;
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = ROTL32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
    };

    h1 ^= len;
    h1 = fmix(h1);
    return h1;
}


uint32_t MurmurHash3_x86_32_positive(const std::string& s, uint32_t seed) {
    uint32_t out = MurmurHash3_x86_32_void(s.c_str(), static_cast<int>(s.length()), seed);
    return out;
}


int32_t MurmurHash3_x86_32(const std::string& s, uint32_t seed) {
    uint32_t outp = MurmurHash3_x86_32_void(s.c_str(), static_cast<int>(s.length()), seed);
    int32_t out;
    *((uint32_t*)(&out)) = outp;
    return out;
}


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_murmurhash3_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator Murmurhash3."
    #else
    R"pbdoc(Implements runtime for operator Murmurhash3. The code is inspired from
`murmur_hash3.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/contrib_ops/cpu//murmur_hash3.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    m.def("MurmurHash3_x86_32_positive", &MurmurHash3_x86_32_positive);
    m.def("MurmurHash3_x86_32", &MurmurHash3_x86_32);
}

#endif
