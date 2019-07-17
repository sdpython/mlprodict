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


class RuntimeSVMRegressor
{
    public:
        
        KERNEL kernel_type_;
        float gamma_;
        float coef0_;
        float degree_;

        // svm_regressor.h
        bool one_class_;
        int64_t feature_count_;
        int64_t vector_count_;
        std::vector<float> rho_;
        std::vector<float> coefficients_;
        std::vector<float> support_vectors_;
        POST_EVAL_TRANSFORM post_transform_;
        SVM_TYPE mode_;  //how are we computing SVM? 0=LibSVC, 1=LibLinear
    
    public:
        
        RuntimeSVMRegressor();
        ~RuntimeSVMRegressor();

        void init(
            py::array_t<float> coefficients,
            py::array_t<float> kernel_params,
            const std::string& kernel_type,
            int64_t n_supports,
            int64_t one_class,
            const std::string& post_transform,
            py::array_t<float> rho,
            py::array_t<float> support_vectors
        );
        
        py::array_t<float> compute(py::array_t<float> X) const;
    
        std::string runtime_options();

        int omp_get_max_threads();

private:

        void Initialize();

        template<typename T>
        float kernel_dot_gil_free(
                const T* A, int64_t a, const std::vector<float>& B,
                int64_t b, int64_t len, KERNEL k) const;
    
        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<float>& X, py::array_t<float>& Z) const;
};


RuntimeSVMRegressor::RuntimeSVMRegressor() {
}


RuntimeSVMRegressor::~RuntimeSVMRegressor() {
}


std::string RuntimeSVMRegressor::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


int RuntimeSVMRegressor::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
}


void RuntimeSVMRegressor::init(
            py::array_t<float> coefficients,
            py::array_t<float> kernel_params,
            const std::string& kernel_type,
            int64_t n_supports,
            int64_t one_class,
            const std::string& post_transform,
            py::array_t<float> rho,
            py::array_t<float> support_vectors
    ) {
    kernel_type_ = to_KERNEL(kernel_type);
    vector_count_ = n_supports;
    array2vector(support_vectors_, support_vectors, float);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    array2vector(rho_, rho, float);
    array2vector(coefficients_, coefficients, float);
    one_class_ = one_class != 0;
        
    std::vector<float> kernel_params_local;
    array2vector(kernel_params_local, kernel_params, float);

    if (!kernel_params_local.empty()) {
      gamma_ = kernel_params_local[0];
      coef0_ = kernel_params_local[1];
      degree_ = kernel_params_local[2];
    }
    else {
      gamma_ = 0.f;
      coef0_ = 0.f;
      degree_ = 0.f;
    }
    
    Initialize();
}


void RuntimeSVMRegressor::Initialize() {
  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size();
    mode_ = SVM_TYPE::SVM_LINEAR;
    kernel_type_ = KERNEL::LINEAR;
  }
}


py::array_t<float> RuntimeSVMRegressor::compute(py::array_t<float> X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");
    // Does not handle 3D tensors
    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
                        
    py::array_t<float> Z(x_dims[0]); // one target only
    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z);
    }
    return Z;
}

template<typename T>
float RuntimeSVMRegressor::kernel_dot_gil_free(
        const T* A, int64_t a,
        const std::vector<float>& B, int64_t b,
        int64_t len, KERNEL k) const {
    double sum = 0;
    const T* pA = A + a;
    const float* pB = B.data() + b;
    if (k == KERNEL::POLY) {
      for (int64_t i = len; i > 0; --i, ++pA, ++pB)
        sum += *pA * *pB;
      sum = gamma_ * sum + coef0_;
      sum = std::pow(sum, degree_);
    } else if (k == KERNEL::SIGMOID) {
      for (int64_t i = len; i > 0; --i, ++pA, ++pB)
        sum += *pA * *pB;
      sum = gamma_ * sum + coef0_;
      sum = std::tanh(sum);
    } else if (k == KERNEL::RBF) {
      for (int64_t i = len; i > 0; --i, ++pA, ++pB) {
        double val = *pA - *pB;
        sum += val * val;
      }
      sum = std::exp(-gamma_ * sum);
    } else if (k == KERNEL::LINEAR) {
      for (int64_t i = len; i > 0; --i, ++pA, ++pB)
        sum += *pA * *pB;
    }
    return (float)sum;
}

    
void RuntimeSVMRegressor::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<float>& X, py::array_t<float>& Z) const {

  auto Z_ = Z.mutable_unchecked<1>();          
  const float* x_data = X.data(0);
  float* z_data = (float*)Z_.data(0);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t n = 0; n < N; ++n) {  //for each example
    int64_t current_weight_0 = n * stride;

    float sum = 0.f;
    if (mode_ == SVM_TYPE::SVM_SVC) {
      for (int64_t j = 0; j < vector_count_; ++j) {
        float val1 = kernel_dot_gil_free(x_data, current_weight_0, support_vectors_,
                                         feature_count_ * j, feature_count_, kernel_type_);
        sum += val1 * coefficients_[j];
      }
      sum += rho_[0];
    } else if (mode_ == SVM_TYPE::SVM_LINEAR) {  //liblinear
      sum = kernel_dot_gil_free(x_data, current_weight_0, coefficients_, 0,
                                feature_count_, kernel_type_);
      sum += rho_[0];
    }
    z_data[n] = (one_class_ && sum > 0) 
                    ? 1.f
                    : (one_class_ ? -1.f : sum);
  }
}

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

    py::class_<RuntimeSVMRegressor> cl (m, "RuntimeSVMRegressor",
        R"pbdoc(Implements runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cl.def(py::init<>());
    cl.def("init", &RuntimeSVMRegressor::init,
           "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cl.def("compute", &RuntimeSVMRegressor::compute,
           "Computes the predictions for the random forest.");
    cl.def("runtime_options", &RuntimeSVMRegressor::runtime_options,
           "Returns indications about how the runtime was compiled.");
    cl.def("omp_get_max_threads", &RuntimeSVMRegressor::omp_get_max_threads,
           "Returns omp_get_max_threads from openmp library.");
}

#endif
