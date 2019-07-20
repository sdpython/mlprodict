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


class RuntimeSVMClassifier
{
    public:
        
        KERNEL kernel_type_;
        float gamma_;
        float coef0_;
        float degree_;

        // svm_classifier.h
        int64_t feature_count_;
        int64_t vector_count_;
        std::vector<float> rho_;
        std::vector<float> coefficients_;
        std::vector<float> support_vectors_;
        POST_EVAL_TRANSFORM post_transform_;
        SVM_TYPE mode_;  //how are we computing SVM? 0=LibSVC, 1=LibLinear
    
        std::vector<float> proba_;
        std::vector<float> probb_;
        bool weights_are_all_positive_;
        std::vector<int64_t> classlabels_int64s_;
        // std::vector<std::string> classlabels_strings_;
    
        int64_t class_count_;
        std::vector<int64_t> vectors_per_class_;
        std::vector<int64_t> starting_vector_;
        
    public:
        
        RuntimeSVMClassifier();
        ~RuntimeSVMClassifier();

        void init(
            py::array_t<int64_t> classlabels_int64s,
            const std::vector<std::string>& classlabels_strings,
            py::array_t<float> coefficients,
            py::array_t<float> kernel_params,
            const std::string& kernel_type,
            const std::string& post_transform,
            py::array_t<float> prob_a,
            py::array_t<float> prob_b,
            py::array_t<float> rho,
            py::array_t<float> support_vectors,
            py::array_t<int64_t> vectors_per_class
        );
        
        py::tuple compute(py::array_t<float> X) const;
    
        std::string runtime_options();

        int omp_get_max_threads();

private:

        void Initialize();

        template<typename T>
        float kernel_dot_gil_free(
                const T* A, int64_t a, const std::vector<float>& B,
                int64_t b, int64_t len, KERNEL k) const;
    
        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<float>& X, py::array_t<int64_t>& Y,
                              py::array_t<float>& Z, int64_t nb_columns) const;
};


RuntimeSVMClassifier::RuntimeSVMClassifier() {
}


RuntimeSVMClassifier::~RuntimeSVMClassifier() {
}


std::string RuntimeSVMClassifier::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


int RuntimeSVMClassifier::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
}


void RuntimeSVMClassifier::init(
            py::array_t<int64_t> classlabels_int64s,
            const std::vector<std::string>& classlabels_strings,
            py::array_t<float> coefficients,
            py::array_t<float> kernel_params,
            const std::string& kernel_type,
            const std::string& post_transform,
            py::array_t<float> prob_a,
            py::array_t<float> prob_b,
            py::array_t<float> rho,
            py::array_t<float> support_vectors,
            py::array_t<int64_t> vectors_per_class
    ) {
    kernel_type_ = to_KERNEL(kernel_type);
    array2vector(support_vectors_, support_vectors, float);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    array2vector(rho_, rho, float);
    array2vector(coefficients_, coefficients, float);
        
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

    array2vector(proba_, prob_a, float);
    array2vector(probb_, prob_b, float);
    array2vector(vectors_per_class_, vectors_per_class, int64_t);
    if (classlabels_strings.size() > 0)
        throw std::runtime_error("This runtime only handles integers.");
    // classlabels_strings_ = classlabels_strings;
    array2vector(classlabels_int64s_, classlabels_int64s, int64_t);
    
    Initialize();
}


void RuntimeSVMClassifier::Initialize() {
  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size();
    mode_ = SVM_TYPE::SVM_LINEAR;
    kernel_type_ = KERNEL::LINEAR;
  }
  
  vector_count_ = 0;
  feature_count_ = 0;
  class_count_ = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(vectors_per_class_.size()); i++) {
    starting_vector_.push_back(vector_count_);
    vector_count_ += vectors_per_class_[i];
  }

  if (classlabels_int64s_.size() > 0) {
    class_count_ = classlabels_int64s_.size();
  } else {
    class_count_ = 1;
  }
  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size() / class_count_;  //liblinear mode
    mode_ = SVM_TYPE::SVM_LINEAR;
    kernel_type_ = KERNEL::LINEAR;
  }
  weights_are_all_positive_ = true;
  for (int64_t i = 0; i < static_cast<int64_t>(coefficients_.size()); i++) {
    if (coefficients_[i] < 0) {
      weights_are_all_positive_ = false;
      break;
    }
  }  
}


int _set_score_svm(int64_t* output_data, float max_weight, const int64_t maxclass,
                   const int64_t n, POST_EVAL_TRANSFORM post_transform_,
                   const std::vector<float>& proba_, bool weights_are_all_positive_,
                   const std::vector<int64_t>& classlabels, int64_t posclass,
                   int64_t negclass) {
  int write_additional_scores = -1;
  if (classlabels.size() == 2) {
    write_additional_scores = post_transform_ == POST_EVAL_TRANSFORM::NONE ? 2 : 0;
    if (proba_.size() == 0) {
      if (weights_are_all_positive_ && max_weight >= 0.5)
        output_data[n] = classlabels[1];
      else if (max_weight > 0 && !weights_are_all_positive_)
        output_data[n] = classlabels[1];
      else
        output_data[n] = classlabels[maxclass];
    } else {
      output_data[n] = classlabels[maxclass];
    }
  } else if (max_weight > 0) {
    output_data[n] = posclass;
  } else {
    output_data[n] = negclass;
  }
  return write_additional_scores;
}


py::tuple RuntimeSVMClassifier::compute(py::array_t<float> X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");
    // Does not handle 3D tensors
    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
    
    int64_t nb_columns = class_count_;
    if (proba_.size() == 0 && vector_count_ > 0) {
    if (class_count_ > 2)
      nb_columns = class_count_ * (class_count_ - 1) / 2;
    else
      nb_columns = 2;
    }

    std::vector<int64_t> dims{N, nb_columns};    
                        
    py::array_t<int64_t> Y(N); // one target only
    py::array_t<float> Z(N * nb_columns); // one target only
    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Y, Z, nb_columns);
    }
    return py::make_tuple(Y, Z);
}

template<typename T>
float RuntimeSVMClassifier::kernel_dot_gil_free(
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


void multiclass_probability(int64_t classcount, const std::vector<float>& r,
                            std::vector<float>& p) {
  int64_t sized2 = classcount * classcount;
  std::vector<float> Q;
  std::vector<float> Qp;
  for (int64_t k = 0; k < sized2; k++) {
    Q.push_back(0);
  }
  for (int64_t k = 0; k < classcount; k++) {
    Qp.push_back(0);
  }
  float eps = 0.005f / static_cast<float>(classcount);
  for (int64_t i = 0; i < classcount; i++) {
    p[i] = 1.0f / static_cast<float>(classcount);  // Valid if k = 1
    for (int64_t j = 0; j < i; j++) {
      Q[i * classcount + i] += r[j * classcount + i] * r[j * classcount + i];
      Q[i * classcount + j] = Q[j * classcount + i];
    }
    for (int64_t j = i + 1; j < classcount; j++) {
      Q[i * classcount + i] += r[j * classcount + i] * r[j * classcount + i];
      Q[i * classcount + j] = -r[j * classcount + i] * r[i * classcount + j];
    }
  }
  for (int64_t loop = 0; loop < 100; loop++) {
    // stopping condition, recalculate QP,pQP for numerical accuracy
    float pQp = 0;
    for (int64_t i = 0; i < classcount; i++) {
      Qp[i] = 0;
      for (int64_t j = 0; j < classcount; j++) {
        Qp[i] += Q[i * classcount + j] * p[j];
      }
      pQp += p[i] * Qp[i];
    }
    float max_error = 0;
    for (int64_t i = 0; i < classcount; i++) {
      float error = std::fabs(Qp[i] - pQp);
      if (error > max_error) {
        max_error = error;
      }
    }
    if (max_error < eps)
      break;

    for (int64_t i = 0; i < classcount; i++) {
      float diff = (-Qp[i] + pQp) / Q[i * classcount + i];
      p[i] += diff;
      pQp = (pQp + diff * (diff * Q[i * classcount + i] + 2 * Qp[i])) / (1 + diff) / (1 + diff);
      for (int64_t j = 0; j < classcount; j++) {
        Qp[j] = (Qp[j] + diff * Q[i * classcount + j]) / (1 + diff);
        p[j] /= (1 + diff);
      }
    }
  }
}


void RuntimeSVMClassifier::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<float>& X,
                py::array_t<int64_t>& Y, py::array_t<float>& Z,
                int64_t nb_columns) const {
  auto Y_ = Y.mutable_unchecked<1>();          
  auto Z_ = Z.mutable_unchecked<1>();          
  const float* x_data = X.data(0);
  int64_t* y_data = (int64_t*)Y_.data(0);
  float* z_data = (float*)Z_.data(0);

  int64_t zindex = 0; 
  for (int64_t n = 0; n < N; n++)  //for each example
  {
    int64_t current_weight_0 = n * stride;
    int64_t maxclass = -1;
    std::vector<float> decisions;
    std::vector<float> scores;
    std::vector<float> kernels;
    std::vector<int64_t> votes;

    if (vector_count_ == 0 && mode_ == SVM_TYPE::SVM_LINEAR) {
      for (int64_t j = 0; j < class_count_; j++) {  //for each class
        auto val = kernel_dot_gil_free(x_data, current_weight_0, coefficients_,
                                       feature_count_ * j,
                                       feature_count_, kernel_type_);
        val += rho_[0];
        scores.push_back(val);
      }
    } else {
      if (vector_count_ == 0)
        throw std::runtime_error("No support vectors.");
      int evals = 0;

      for (int64_t j = 0; j < vector_count_; j++) {
        auto val = kernel_dot_gil_free(x_data, current_weight_0, support_vectors_,
                                       feature_count_ * j,
                                       feature_count_, kernel_type_);
        kernels.push_back(val);
      }
      votes.resize(class_count_, 0);
      for (int64_t i = 0; i < class_count_; i++) {        // for each class
        for (int64_t j = i + 1; j < class_count_; j++) {  // for each class
          double sum = 0;
          int64_t start_index_i = starting_vector_[i];  // *feature_count_;
          int64_t start_index_j = starting_vector_[j];  // *feature_count_;

          int64_t class_i_support_count = vectors_per_class_[i];
          int64_t class_j_support_count = vectors_per_class_[j];

          int64_t pos1 = (vector_count_) * (j - 1);
          int64_t pos2 = (vector_count_) * (i);
          const float* val1 = &(coefficients_[pos1 + start_index_i]);
          const float* val2 = &(kernels[start_index_i]);
          for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          val1 = &(coefficients_[pos2 + start_index_j]);
          val2 = &(kernels[start_index_j]);
          for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          sum += rho_[evals];
          scores.push_back((float)sum);
          ++(votes[sum > 0 ? i : j]);
          ++evals;  //index into rho
        }
      }
    }

    if (proba_.size() > 0 && mode_ == SVM_TYPE::SVM_SVC) {
      //compute probabilities from the scores
      int64_t num = class_count_ * class_count_;
      std::vector<float> probsp2(num, 0.f);
      std::vector<float> estimates(class_count_, 0.f);
      int64_t index = 0;
      for (int64_t i = 0; i < class_count_; ++i) {
        int64_t p1 = i * class_count_ + i + 1;
        int64_t p2 = (i + 1) * class_count_ + i;
        for (int64_t j = i + 1; j < class_count_; ++j, ++index) {
          float val1 = sigmoid_probability(scores[index], proba_[index], probb_[index]);
          float val2 = std::max(val1, 1.0e-7f);
          val2 = std::min(val2, 1 - 1.0e-7f);
          probsp2[p1] = val2;
          probsp2[p2] = 1 - val2;
          ++p1;
          p2 += class_count_;
        }
      }
      multiclass_probability(class_count_, probsp2, estimates);
      // copy probabilities back into scores
      scores.resize(estimates.size());
      std::copy(estimates.begin(), estimates.end(), scores.begin());
    }

    float max_weight = 0;
    if (votes.size() > 0) {
      auto it_maxvotes = std::max_element(votes.begin(), votes.end());
      maxclass = std::distance(votes.begin(), it_maxvotes);
    } else {
      auto it_max_weight = std::max_element(scores.begin(), scores.end());
      maxclass = std::distance(scores.begin(), it_max_weight);
      max_weight = *it_max_weight;
    }

    // write top class
    // onnx specs expects one column per class.
    int write_additional_scores = -1;
    if (rho_.size() == 1) {
      write_additional_scores = _set_score_svm(
          y_data, max_weight, maxclass, n, post_transform_, proba_,
          weights_are_all_positive_, classlabels_int64s_, 1, 0);
    } else if (classlabels_int64s_.size() > 0) {  //multiclass
        y_data[n] = classlabels_int64s_[maxclass];
    } else {
        y_data[n] = maxclass;
    }

    write_scores(scores, post_transform_, z_data + zindex, write_additional_scores);
    zindex += scores.size();
  }
}

#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_svm_classifier_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator SVMClassifier."
    #else
    R"pbdoc(Implements runtime for operator SVMClassifier. The code is inspired from
`svm_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeSVMClassifier> cl (m, "RuntimeSVMClassifier",
        R"pbdoc(Implements runtime for operator SVMClassifier. The code is inspired from
`svm_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cl.def(py::init<>());
    cl.def("init", &RuntimeSVMClassifier::init,
           "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cl.def("compute", &RuntimeSVMClassifier::compute,
           "Computes the predictions for the SVM classifier.");
    cl.def("runtime_options", &RuntimeSVMClassifier::runtime_options,
           "Returns indications about how the runtime was compiled.");
    cl.def("omp_get_max_threads", &RuntimeSVMClassifier::omp_get_max_threads,
           "Returns omp_get_max_threads from openmp library.");
}

#endif
