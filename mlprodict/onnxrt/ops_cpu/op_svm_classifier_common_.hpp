// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <thread>
#include <iterator>

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
#include "op_common_num_.hpp"


template<typename NTYPE>
class RuntimeSVMCommon
{
    public:

        KERNEL kernel_type_;
        NTYPE gamma_;
        NTYPE coef0_;
        int64_t degree_;

        // svm_regressor.h
        int64_t feature_count_;
        int64_t vector_count_;
        std::vector<NTYPE> rho_;
        std::vector<NTYPE> coefficients_;
        std::vector<NTYPE> support_vectors_;
        POST_EVAL_TRANSFORM post_transform_;
        SVM_TYPE mode_;  //how are we computing SVM? 0=LibSVC, 1=LibLinear
        int omp_N_;
    
    public:

        RuntimeSVMCommon(int omp_N) { omp_N_ = omp_N; }
        ~RuntimeSVMCommon() { }
        
        void init(py::array_t<NTYPE> coefficients,
                  py::array_t<NTYPE> kernel_params,
                  const std::string& kernel_type,
                  const std::string& post_transform,
                  py::array_t<NTYPE> rho,
                  py::array_t<NTYPE> support_vectors);
                    

        NTYPE kernel_dot_gil_free(
                const NTYPE* A, int64_t a, const std::vector<NTYPE>& B,
                int64_t b, int64_t len, KERNEL k) const;
    
    public:
        
        std::string runtime_options();

        int omp_get_max_threads();
    
};


template<typename NTYPE>
void RuntimeSVMCommon<NTYPE>::init(
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            const std::string& post_transform,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors
    ) {
    kernel_type_ = to_KERNEL(kernel_type);
    array2vector(support_vectors_, support_vectors, NTYPE);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    array2vector(rho_, rho, NTYPE);
    array2vector(coefficients_, coefficients, NTYPE);
        
    std::vector<NTYPE> kernel_params_local;
    array2vector(kernel_params_local, kernel_params, NTYPE);

    if (!kernel_params_local.empty()) {
        gamma_ = kernel_params_local[0];
        coef0_ = kernel_params_local[1];
        degree_ = static_cast<int64_t>(kernel_params_local[2]);
    }
    else {
        gamma_ = (NTYPE)0;
        coef0_ = (NTYPE)0;
        degree_ = 0;
    }
}


template<typename NTYPE>
NTYPE RuntimeSVMCommon<NTYPE>::kernel_dot_gil_free(
        const NTYPE* A, int64_t a,
        const std::vector<NTYPE>& B, int64_t b,
        int64_t len, KERNEL k) const {
    double sum = 0;
    double val;
    const NTYPE* pA = A + a;
    const NTYPE* pB = B.data() + b;
    switch(k) {
        case KERNEL::POLY:
            sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
            sum = gamma_ * sum + coef0_;
            switch (degree_) {
                case 2:
                    sum = sum * sum;
                    break;
                case 3:
                    sum = sum * sum * sum;
                    break;
                case 4:
                    val = sum * sum;
                    sum = val * val;
                    break;
                default:
                    sum = std::pow(sum, degree_);
                    break;
            }
            break;
        case KERNEL::SIGMOID:
            sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
            sum = gamma_ * sum + coef0_;
            sum = std::tanh(sum);
            break;
        case KERNEL::RBF:
            for (int64_t i = len; i > 0; --i, ++pA, ++pB) {
                val = *pA - *pB;
                sum += val * val;
            }
            sum = std::exp(-gamma_ * sum);
            break;
        case KERNEL::LINEAR:
            sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
            break;
    }
    return (NTYPE)sum;
}




template<typename NTYPE>
std::string RuntimeSVMCommon<NTYPE>::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


template<typename NTYPE>
int RuntimeSVMCommon<NTYPE>::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
}

py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(py::array_t<float>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(py::array_t<double>& Z) {
    return Z.mutable_unchecked<1>();
}

