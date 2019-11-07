// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

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
class RuntimeSVMRegressor
{
    public:
        
        KERNEL kernel_type_;
        NTYPE gamma_;
        NTYPE coef0_;
        NTYPE degree_;

        // svm_regressor.h
        bool one_class_;
        int64_t feature_count_;
        int64_t vector_count_;
        std::vector<NTYPE> rho_;
        std::vector<NTYPE> coefficients_;
        std::vector<NTYPE> support_vectors_;
        POST_EVAL_TRANSFORM post_transform_;
        SVM_TYPE mode_;  //how are we computing SVM? 0=LibSVC, 1=LibLinear
    
    public:
        
        RuntimeSVMRegressor();
        ~RuntimeSVMRegressor();

        void init(
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            int64_t n_supports,
            int64_t one_class,
            const std::string& post_transform,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors
        );
        
        py::array_t<NTYPE> compute(py::array_t<NTYPE> X) const;
    
        std::string runtime_options();

        int omp_get_max_threads();

private:

        void Initialize();

        NTYPE kernel_dot_gil_free(
                const NTYPE* A, int64_t a, const std::vector<NTYPE>& B,
                int64_t b, int64_t len, KERNEL k) const;
    
        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z) const;
};


template<typename NTYPE>
RuntimeSVMRegressor<NTYPE>::RuntimeSVMRegressor() {
}


template<typename NTYPE>
RuntimeSVMRegressor<NTYPE>::~RuntimeSVMRegressor() {
}


template<typename NTYPE>
std::string RuntimeSVMRegressor<NTYPE>::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


template<typename NTYPE>
int RuntimeSVMRegressor<NTYPE>::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
}


template<typename NTYPE>
void RuntimeSVMRegressor<NTYPE>::init(
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            int64_t n_supports,
            int64_t one_class,
            const std::string& post_transform,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors
    ) {
    kernel_type_ = to_KERNEL(kernel_type);
    vector_count_ = n_supports;
    array2vector(support_vectors_, support_vectors, NTYPE);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    array2vector(rho_, rho, NTYPE);
    array2vector(coefficients_, coefficients, NTYPE);
    one_class_ = one_class != 0;
        
    std::vector<NTYPE> kernel_params_local;
    array2vector(kernel_params_local, kernel_params, NTYPE);

    if (!kernel_params_local.empty()) {
      gamma_ = kernel_params_local[0];
      coef0_ = kernel_params_local[1];
      degree_ = kernel_params_local[2];
    }
    else {
      gamma_ = (NTYPE)0;
      coef0_ = (NTYPE)0;
      degree_ = (NTYPE)0;
    }
    
    Initialize();
}


template<typename NTYPE>
void RuntimeSVMRegressor<NTYPE>::Initialize() {
  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size();
    mode_ = SVM_TYPE::SVM_LINEAR;
    kernel_type_ = KERNEL::LINEAR;
  }
}


template<typename NTYPE>
py::array_t<NTYPE> RuntimeSVMRegressor<NTYPE>::compute(py::array_t<NTYPE> X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");
    // Does not handle 3D tensors
    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
                        
    py::array_t<NTYPE> Z(x_dims[0]); // one target only
    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z);
    }
    return Z;
}

template<typename NTYPE>
NTYPE RuntimeSVMRegressor<NTYPE>::kernel_dot_gil_free(
        const NTYPE* A, int64_t a,
        const std::vector<NTYPE>& B, int64_t b,
        int64_t len, KERNEL k) const {
    double sum = 0;
    const NTYPE* pA = A + a;
    const NTYPE* pB = B.data() + b;
    if (k == KERNEL::POLY) {
      sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
      sum = gamma_ * sum + coef0_;
      if (degree_ == 2)
        sum = sum * sum;
      else if (degree_ == 3)
        sum = sum * sum * sum;
      else if (degree_ == 4) {
        double s2 = sum * sum;
        sum = s2 * s2;
      }
      else
        sum = std::pow(sum, degree_);
    } else if (k == KERNEL::SIGMOID) {
      sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
      sum = gamma_ * sum + coef0_;
      sum = std::tanh(sum);
    } else if (k == KERNEL::RBF) {
      for (int64_t i = len; i > 0; --i, ++pA, ++pB) {
        double val = *pA - *pB;
        sum += val * val;
      }
      sum = std::exp(-gamma_ * sum);
    } else if (k == KERNEL::LINEAR) {
      sum = vector_dot_product_pointer_sse(pA, pB, (size_t)len);
    }
    return (NTYPE)sum;
}

    
py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(py::array_t<float>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(py::array_t<double>& Z) {
    return Z.mutable_unchecked<1>();
}


template<typename NTYPE>
void RuntimeSVMRegressor<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z) const {

  auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
  const NTYPE* x_data = X.data(0);
  NTYPE* z_data = (NTYPE*)Z_.data(0);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t n = 0; n < N; ++n) {  //for each example
    int64_t current_weight_0 = n * stride;

    NTYPE sum = (NTYPE)0;
    if (mode_ == SVM_TYPE::SVM_SVC) {
      for (int64_t j = 0; j < vector_count_; ++j) {
        NTYPE val1 = kernel_dot_gil_free(x_data, current_weight_0, support_vectors_,
                                         feature_count_ * j, feature_count_, kernel_type_);
        sum += coefficients_[j] * val1;
      }
      sum += rho_[0];
    } else if (mode_ == SVM_TYPE::SVM_LINEAR) {  //liblinear
      sum = kernel_dot_gil_free(x_data, current_weight_0, coefficients_, 0,
                                feature_count_, kernel_type_);
      sum += rho_[0];
    }
    z_data[n] = one_class_ ? (sum > 0 ? 1 : -1) : sum;
  }
}

class RuntimeSVMRegressorFloat : public RuntimeSVMRegressor<float>
{
    public:
        RuntimeSVMRegressorFloat() : RuntimeSVMRegressor<float>() {}
};


class RuntimeSVMRegressorDouble : public RuntimeSVMRegressor<double>
{
    public:
        RuntimeSVMRegressorDouble() : RuntimeSVMRegressor<double>() {}
};




#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_svm_regressor_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator SVMRegressor."
    #else
    R"pbdoc(Implements runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeSVMRegressorFloat> clf (m, "RuntimeSVMRegressorFloat",
        R"pbdoc(Implements float runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &RuntimeSVMRegressorFloat::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    clf.def("compute", &RuntimeSVMRegressorFloat::compute,
            "Computes the predictions for the SVM regressor.");
    clf.def("runtime_options", &RuntimeSVMRegressorFloat::runtime_options,
            "Returns indications about how the runtime was compiled.");
    clf.def("omp_get_max_threads", &RuntimeSVMRegressorFloat::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    py::class_<RuntimeSVMRegressorDouble> cld (m, "RuntimeSVMRegressorDouble",
        R"pbdoc(Implements Double runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &RuntimeSVMRegressorDouble::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cld.def("compute", &RuntimeSVMRegressorDouble::compute,
            "Computes the predictions for the SVM regressor.");
    cld.def("runtime_options", &RuntimeSVMRegressorDouble::runtime_options,
            "Returns indications about how the runtime was compiled.");
    cld.def("omp_get_max_threads", &RuntimeSVMRegressorDouble::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");
}

#endif
