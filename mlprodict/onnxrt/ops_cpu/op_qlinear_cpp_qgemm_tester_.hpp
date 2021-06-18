#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/providers/cpu/nn/qlinearconv_op_test.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#if !defined(UNUSED_VARIABLE)
#if defined(__GNUC__)
# define UNUSED_VARIABLE __attribute__((unused))
#else
# define UNUSED_VARIABLE
#endif
#endif

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

#include "op_conv_matrices_.hpp"


template <typename T>
class MatrixGuardBuffer {
public:
    MatrixGuardBuffer() {
        _BaseBuffer = nullptr;
        _BaseBufferSize = 0;
        _ElementsAllocated = 0;
    }

    ~MatrixGuardBuffer(void) { ReleaseBuffer(); }

    T* GetBuffer(size_t Elements, bool ZeroFill = false) {
        // Check if the internal buffer needs to be reallocated.

        if (Elements > _ElementsAllocated) {
            ReleaseBuffer();

            // Reserve a virtual address range for the allocation plus an unmapped
            // guard region.

            constexpr size_t BufferAlignment = 64 * 1024;
            constexpr size_t GuardPadding = 256 * 1024;
            size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);
            _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
            _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
            _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

            if (_BaseBuffer == nullptr)
                abort();

            // Commit the number of bytes for the allocation leaving the upper
            // guard region as unmapped.

#if defined(_WIN32)
            if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr)
                throw std::bad_alloc();
#else
            if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0)
                abort();
#endif
            _ElementsAllocated = BytesToAllocate / sizeof(T);
            _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
        }

        //
        //
        //

        T* GuardAddress = _GuardAddress;
        T* buffer = GuardAddress - Elements;

        if (ZeroFill) {
            std::fill_n(buffer, Elements, T(0));
        }
        else {
            const int MinimumFillValue = -23;
            const int MaximumFillValue = 23;
            int FillValue = MinimumFillValue;
            T* FillAddress = buffer;

            while (FillAddress < GuardAddress) {
                *FillAddress++ = (T)FillValue;
                FillValue++;
                if (FillValue > MaximumFillValue)
                    FillValue = MinimumFillValue;
            }
        }

        return buffer;
    }

    void ReleaseBuffer(void) {
        if (_BaseBuffer != nullptr) {
#if defined(_WIN32)
            VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
            munmap(_BaseBuffer, _BaseBufferSize);
#endif
            _BaseBuffer = nullptr;
            _BaseBufferSize = 0;
        }

        _ElementsAllocated = 0;
    }

private:
    size_t _ElementsAllocated;
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    T* _GuardAddress;
};


void TestLocalGemm(
    bool TransA, bool TransB,
    size_t M, size_t N, size_t K,
    float alpha, const float* A, size_t lda,
    const float* B, size_t ldb, float beta,
    float* C, size_t ldc);


struct GEMM_U8X8_SHAPE_PARAMS {
    size_t M = 0;
    size_t N = 0;
    size_t K = 0;
    bool BIsSigned = false;
};


struct GEMM_U8X8_DATA_PARAMS {
    const uint8_t* A = nullptr;
    size_t lda = 0;
    uint8_t ZeroPointA = 0;
    const void* B = 0;
    size_t ldb = 0;
    const uint8_t* ZeroPointB = nullptr;
    bool BIsPacked = false;
    bool PerColumnZeroPoints = false;
    int32_t* C = nullptr;
    size_t ldc = 0;
};

template <typename T>
void TestLocalQGemmBatch(
    const GEMM_U8X8_SHAPE_PARAMS& sh,
    const GEMM_U8X8_DATA_PARAMS* dps,
    size_t BatchN) {
    for (size_t i = 0; i < BatchN; ++i) {
        const GEMM_U8X8_DATA_PARAMS& dp = dps[i];
        QGemm<uint8_t, T, int32_t>(
            false, false, sh.M, sh.N, sh.K, (int32_t)1, dp.A, (const T*)dp.B, (int32_t)0, dp.C, dp.lda, dp.ldb, dp.ldc,
            dp.ZeroPointA, (const T*)dp.ZeroPointB, dp.BIsPacked, dp.PerColumnZeroPoints);
    }
}


template <typename T>
inline void TestQGemmBatch(
    const GEMM_U8X8_SHAPE_PARAMS& Shape,
    const GEMM_U8X8_DATA_PARAMS* DataParams,
    size_t BatchN) {
    TestLocalQGemmBatch<T>(Shape, DataParams, BatchN);
}


class QgemmTestBase {
public:
    virtual ~QgemmTestBase(void) {}
    virtual void ExecuteShort(void) {}
    virtual void ExecuteLong(void) {}
};


template<typename T>
class QgemmU8X8U8X8TestBase : public QgemmTestBase {
protected:
    QgemmU8X8U8X8TestBase() {}

    void TestGemm(
        size_t M, size_t N, size_t K, size_t BatchSize,
        const uint8_t* A, size_t lda, uint8_t offa,
        const uint8_t* B, size_t ldb, uint8_t offb,
        bool BIsSigned, int32_t* C, size_t ldc) {
        GEMM_U8X8_SHAPE_PARAMS GemmShape;
        GemmShape.M = M;
        GemmShape.N = N;
        GemmShape.K = K;
        GemmShape.BIsSigned = BIsSigned;

        std::vector<GEMM_U8X8_DATA_PARAMS> GemmParameters(BatchSize);

        for (size_t i = 0; i < GemmParameters.size(); i++) {
            auto& params = GemmParameters[i];
            params.A = A + (M * K * i);
            params.lda = lda;
            params.ZeroPointA = offa;
            params.ZeroPointB = &offb;
            params.C = C + (M * N * i);
            params.ldc = ldc;
            params.B = B + (K * N * i);
            params.ldb = ldb;
        }
        TestQGemmBatch<T>(GemmShape, GemmParameters.data(), BatchSize);
    }

    void TestGemm(
        size_t M, size_t N, size_t K, size_t BatchSize,
        const uint8_t* A, size_t lda, uint8_t offa,
        const uint8_t* B, size_t ldb, const uint8_t* offb,
        bool BIsSigned, int32_t* C, size_t ldc,
        bool PerColumnZeroPoints = false) {
        GEMM_U8X8_SHAPE_PARAMS GemmShape;
        GemmShape.M = M;
        GemmShape.N = N;
        GemmShape.K = K;
        GemmShape.BIsSigned = BIsSigned;

        std::vector<GEMM_U8X8_DATA_PARAMS> GemmParameters(BatchSize);

        for (size_t i = 0; i < GemmParameters.size(); i++) {
            auto& params = GemmParameters[i];
            params.A = A + M * K * i;
            params.lda = lda;
            params.ZeroPointA = offa;
            params.ZeroPointB = offb;
            params.PerColumnZeroPoints = PerColumnZeroPoints;
            params.C = C + M * N * i;
            params.ldc = ldc;
            params.B = B + K * N * i;
            params.ldb = ldb;
        }
        TestQGemmBatch<T>(GemmShape, GemmParameters.data(), BatchSize);
    }
};

template <typename xint8_t, typename OutputType>
class QgemmU8X8Test;

template <typename xint8_t>
class QgemmU8X8Test<xint8_t, int32_t> : public QgemmU8X8U8X8TestBase<xint8_t> {
public:
    void Test(size_t M, size_t N, size_t K, size_t BatchSize, uint8_t offa, uint8_t offb) {
        const uint8_t* A = BufferA.GetBuffer(K * M * BatchSize);
        const uint8_t* B = BufferB.GetBuffer(N * K * BatchSize);
        int32_t* C = BufferC.GetBuffer(N * M * BatchSize);
        int32_t* CReference = BufferCReference.GetBuffer(N * M * BatchSize);
        Test(M, N, K, BatchSize, A, K, offa, B, N, offb, C, CReference, N);
    }

    void Test(size_t M, size_t N, size_t K, size_t BatchSize, uint8_t offa) {
        const uint8_t* A = BufferA.GetBuffer(K * M * BatchSize);
        const uint8_t* B = BufferB.GetBuffer(N * K * BatchSize);
        const uint8_t* ZeroPointB = BufferZeroPointB.GetBuffer(N);
        int32_t* C = BufferC.GetBuffer(N * M * BatchSize);
        int32_t* CReference = BufferCReference.GetBuffer(N * M * BatchSize);
        Test(M, N, K, BatchSize, A, K, offa, B, N, ZeroPointB, C, CReference, N);
    }

    void Test(
        size_t M, size_t N, size_t K, size_t BatchSize,
        const uint8_t* A, size_t lda, uint8_t offa,
        const uint8_t* B, size_t ldb, uint8_t offb,
        int32_t* C, int32_t* CReference, size_t ldc) {
        std::fill_n(C, M * N * BatchSize, -1);
        std::fill_n(CReference, M * N * BatchSize, -1);

        this->TestGemm(M, N, K, BatchSize, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc);
        ReferenceQgemm(M, N, K, BatchSize, A, lda, offa, (const xint8_t*)B, ldb,
            (xint8_t)offb, CReference, ldc);

        for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++, f++) {
                    if (C[f] != CReference[f])
                        throw std::invalid_argument(MakeString(
                            C[f], "!=", CReference[f], "@[", batch, "x", m, "x", n, "], ",
                            "Batch=", BatchSize, "M=", M, ", N=", N, ", K=", K,
                            ", offa=", int(offa), ", offb=", int(offb)));
                }
            }
        }
    }

    void Test(
        size_t M, size_t N, size_t K, size_t BatchSize,
        const uint8_t* A, size_t lda, uint8_t offa,
        const uint8_t* B, size_t ldb, const uint8_t* offb,
        int32_t* C, int32_t* CReference, size_t ldc,
        bool PerColumnZeroPoints) {
        std::fill_n(C, M * N * BatchSize, -1);
        std::fill_n(CReference, M * N * BatchSize, -1);

        this->TestGemm(M, N, K, BatchSize, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc);
        ReferenceQgemm(M, N, K, BatchSize, A, lda, offa, (const xint8_t*)B, ldb,
            (const xint8_t*)offb, CReference, ldc, PerColumnZeroPoints);

        for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++, f++) {
                    if (C[f] != CReference[f])
                        throw std::invalid_argument(MakeString(
                            C[f], "!=", CReference[f], "@[", batch, "x", m, "x", n, "], "
                            , "Batch=", BatchSize, "M=", M, ", N=", N, ", K=", K
                            , ", offa=", int(offa), ", offb[0]=", int(offb[0])));
                }
            }
        }
    }

private:
    void ReferenceQgemm(
        size_t M, size_t N, size_t K, size_t BatchSize,
        const uint8_t* A, size_t lda, uint8_t offa,
        const xint8_t* B, size_t ldb, xint8_t offb,
        int32_t* C, size_t ldc) {
        for (size_t batch = 0; batch < BatchSize; batch++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    const uint8_t* a = A + (M * K * batch) + (m * lda);
                    const xint8_t* b = B + (K * N * batch) + n;
                    int32_t* c = C + (M * N * batch) + (m * ldc) + n;
                    int32_t sum = 0;
                    for (size_t k = 0; k < K; k++) {
                        sum += ((int32_t(*b) - offb) * (int32_t(*a) - offa));
                        b += ldb;
                        a += 1;
                    }
                    *c = sum;
                }
            }
        }
    }

    void ReferenceQgemm(
        size_t M, size_t N, size_t K, size_t BatchSize,
        const uint8_t* A, size_t lda, uint8_t offa,
        const xint8_t* B, size_t ldb, const xint8_t* offb,
        int32_t* C, size_t ldc, bool PerColumnZeroPoints) {
        for (size_t batch = 0; batch < BatchSize; batch++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    const uint8_t* a = A + (M * K * batch) + (m * lda);
                    const xint8_t* b = B + (K * N * batch) + n;
                    int32_t* c = C + (M * N * batch) + (m * ldc) + n;
                    int32_t sum = 0;
                    xint8_t of = PerColumnZeroPoints ? offb[n] : offb[0];
                    for (size_t k = 0; k < K; k++) {
                        sum += ((int32_t(*b) - of) * (int32_t(*a) - offa));
                        b += ldb;
                        a += 1;
                    }
                    *c = sum;
                }
            }
        }
    }

    MatrixGuardBuffer<uint8_t> BufferA;
    MatrixGuardBuffer<uint8_t> BufferB;
    MatrixGuardBuffer<uint8_t> BufferZeroPointB;
    MatrixGuardBuffer<int32_t> BufferC;
    MatrixGuardBuffer<int32_t> BufferCReference;
    const bool BIsSigned = std::is_signed<xint8_t>::value;

public:

    void ExecuteLong(void) override {
        int count = 0;
        int cstmn = 96;
        static const uint8_t zero_points[] = { 0, 18, /*75,*/ 128, /*157, 231,*/ 255 };

#ifdef _DEBUG
        printf("next B\n");
#endif
        for (size_t M = 1; M < 160; M += 13) {
            for (size_t N = 1; N < 160; N += 13) {
                for (size_t K = 1; K < 160; K += 13) {
                    Test(M, N, K, 1, 18, 24);
                }
            }
#ifdef _DEBUG
            printf("M %zd\n", M);
#endif
        }

#ifdef _DEBUG
        printf("next C\n");
#endif
        for (size_t M = 160; M < 320; M += 24) {
            for (size_t N = 112; N < 320; N += 24) {
                for (size_t K = 1; K < 16; K += 4) {
                    Test(M, N, K, 1, 1, 3);
                }
                for (size_t K = 16; K < 160; K += 64) {
                    Test(M, N, K, 1, 5, 7);
                }
            }
#ifdef _DEBUG
            printf("M %zd\n", M);
#endif
        }

#ifdef _DEBUG
        printf("next A\n");
#endif
        for (size_t a = 0; a < _countof(zero_points); a++) {
            uint8_t offa = zero_points[a];

            for (size_t b = 0; b < _countof(zero_points); b++) {
                uint8_t offb = zero_points[b];

                for (size_t M = 16; M < 160; M += cstmn) {
                    for (size_t N = 16; N < 160; N += cstmn) {
                        static const size_t ks[] = {
                            1, 2, 3, /*4, 5, 6, 7, 8, 9, 10,*/ 16, 20,
                            32, /*48, 64, 118, 119, 120, 121,*/ 122,
                            160/*, 240, 320*/ };
                        for (size_t k = 0; k < _countof(ks); k++) {
                            size_t K = ks[k];

                            Test(M, N, K, 1, offa, offb);
                            ++count;
                            Test(M, N + 1, K, 1, offa, offb);
                            ++count;
                            Test(M + 1, N, K, 1, offa, offb);
                            ++count;
                            Test(M + 1, N + 1, K, 1, offa, offb);
                            ++count;
                            Test(M + 3, N + 2, K, 1, offa, offb);
                            ++count;
                            Test(M + 4, N, K, 1, offa, offb);
                            ++count;
                            Test(M, N + 4, K, 1, offa, offb);
                            ++count;
                            Test(M + 4, N + 4, K, 1, offa, offb);
                            ++count;
                            Test(M + 3, N + 7, K, 1, offa, offb);
                            ++count;
                            Test(M + 8, N, K, 1, offa, offb);
                            ++count;
                            Test(M, N + 8, K, 1, offa, offb);
                            ++count;
                            Test(M + 12, N + 12, K, 1, offa, offb);
                            ++count;
                            Test(M + 13, N, K, 1, offa, offb);
                            ++count;
                            Test(M, N + 15, K, 1, offa, offb);
                            ++count;
                            Test(M + 15, N + 15, K, 1, offa, offb);
                            ++count;

                            Test(M, N, K, 7 + a, offa, offb);
                            ++count;
                            Test(M + 3, N, K, 7 + a, offa, offb);
                            ++count;
                            Test(M, N + 1, K, 7 + a, offa, offb);
                            ++count;
                            Test(M + 12, N, K, 7 + a, offa, offb);
                            ++count;
                            Test(M, N + 15, K, 7 + a, offa, offb);
                            ++count;
                            Test(M + 15, N + 15, K, 7 + a, offa, offb);
                            ++count;
                        }
                    }
#ifdef _DEBUG
                    printf("count=%d a %zd/%zd b %zd/%zd M %zd\n", count, a, _countof(zero_points), b, _countof(zero_points), M);
#endif
                    if (count == 0)
                        throw std::length_error("No test were run.");
                }
            }
        }

#ifdef _DEBUG
        printf("next D\n");
#endif
    }
};

template <typename xint8_t>
class QgemmU8X8Test<xint8_t, float> : public QgemmU8X8U8X8TestBase<xint8_t> {
public:
    void Test(size_t M, size_t N, size_t K, size_t BatchSize, uint8_t offa, uint8_t offb) {
        const uint8_t* A = BufferA.GetBuffer(K * M * BatchSize);
        const uint8_t* B = BufferB.GetBuffer(N * K * BatchSize);
        float* C = BufferC.GetBuffer(N * M * BatchSize);
        float* CReference = BufferCReference.GetBuffer(N * M * BatchSize);
        const float* Bias = BufferBias.GetBuffer(N);

        const float AScale = 0.5f;
        float* AFloat = BufferAFloat.GetBuffer(K * M * BatchSize);
        for (size_t b = 0; b < BatchSize; b++) {
            DequantizeLinear(A + K * M * b, AFloat + K * M * b, K * M, AScale, offa);
        }

        const float BScale = 0.25f;
        float* BFloat = BufferBFloat.GetBuffer(N * K * BatchSize);
        for (size_t b = 0; b < BatchSize; b++) {
            DequantizeLinear((xint8_t*)(B + N * K * b), BFloat + N * K * b, N * K, BScale, xint8_t(offb));
        }

        const float CScale = AScale * BScale;

        Test(M, N, K, BatchSize, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, nullptr);
        Test(M, N, K, BatchSize, A, AFloat, K, offa, B, BFloat, N, offb, C, CReference, N, CScale, Bias);
    }

    void Test(
        size_t M, size_t N, size_t K, size_t BatchSize, const uint8_t* A,
        const float* AFloat, size_t lda, uint8_t offa,
        const uint8_t* B, const float* BFloat, size_t ldb, uint8_t offb,
        float* C, float* CReference, size_t ldc, float CScale, const float* Bias) {
        for (size_t b = 0; b < BatchSize; b++) {
            TestLocalGemm(
                false, false, M, N, K, 1.0f,
                AFloat + K * M * b, lda,
                BFloat + N * K * b, ldb, 0.0f,
                CReference + N * M * b, ldc);
        }

        if (Bias != nullptr) {
            for (size_t b = 0; b < BatchSize; b++) {
                for (size_t m = 0; m < M; m++) {
                    for (size_t n = 0; n < N; n++) {
                        CReference[N * M * b + m * ldc + n] += Bias[n];
                    }
                }
            }
        }

        this->TestGemm(M, N, K, BatchSize, A, lda, offa, B, ldb, offb, BIsSigned, C, ldc, CScale, Bias);

        for (size_t batch = 0, f = 0; batch < BatchSize; batch++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++, f++) {
                    // Sensitive to comparing positive/negative zero.
                    if (C[f] != CReference[f])
                        throw std::invalid_argument(MakeString(
                            C[f], "!=", CReference[f], "@[", batch, "x", m, "x", n, "], "
                            , "Batch=", BatchSize, "M=", M, ", N=", N, ", K=", K
                            , ", offa=", int(offa), ", offb=", int(offb)));
                }
            }
        }
    }

private:
    template <typename qint8_t>
    void DequantizeLinear(const qint8_t* Input,
        float* Output,
        size_t N,
        float scale,
        qint8_t offset) {
        for (size_t n = 0; n < N; n++) {
            Output[n] = float((int32_t(Input[n]) - offset)) * scale;
        }
    }

    MatrixGuardBuffer<uint8_t> BufferA;
    MatrixGuardBuffer<uint8_t> BufferB;
    MatrixGuardBuffer<float> BufferAFloat;
    MatrixGuardBuffer<float> BufferBFloat;
    MatrixGuardBuffer<float> BufferC;
    MatrixGuardBuffer<float> BufferCReference;
    MatrixGuardBuffer<float> BufferBias;
    const bool BIsSigned = std::is_signed<xint8_t>::value;
};

void TestQGemm0();

void TestQGemm1();
