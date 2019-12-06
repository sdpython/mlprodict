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
class RuntimeSVMClassifier
{
    public:
        
        KERNEL kernel_type_;
        NTYPE gamma_;
        NTYPE coef0_;
        NTYPE degree_;

        // svm_classifier.h
        int64_t feature_count_;
        int64_t vector_count_;
        std::vector<NTYPE> rho_;
        std::vector<NTYPE> coefficients_;
        std::vector<NTYPE> support_vectors_;
        POST_EVAL_TRANSFORM post_transform_;
        SVM_TYPE mode_;  //how are we computing SVM? 0=LibSVC, 1=LibLinear
    
        std::vector<NTYPE> proba_;
        std::vector<NTYPE> probb_;
        bool weights_are_all_positive_;
        std::vector<int64_t> classlabels_ints_;
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
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            const std::string& post_transform,
            py::array_t<NTYPE> prob_a,
            py::array_t<NTYPE> prob_b,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors,
            py::array_t<int64_t> vectors_per_class
        );
        
        py::tuple compute(py::array_t<NTYPE> X) const;
    
        std::string runtime_options();

        int omp_get_max_threads();

private:

        void Initialize();

        NTYPE kernel_dot_gil_free(
                const NTYPE* A, int64_t a, const std::vector<NTYPE>& B,
                int64_t b, int64_t len, KERNEL k) const;
    
        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<int64_t>& Y,
                              py::array_t<NTYPE>& Z, int64_t nb_columns) const;
};


template<typename NTYPE>
RuntimeSVMClassifier<NTYPE>::RuntimeSVMClassifier() {
}


template<typename NTYPE>
RuntimeSVMClassifier<NTYPE>::~RuntimeSVMClassifier() {
}


template<typename NTYPE>
std::string RuntimeSVMClassifier<NTYPE>::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


template<typename NTYPE>
int RuntimeSVMClassifier<NTYPE>::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
}


template<typename NTYPE>
void RuntimeSVMClassifier<NTYPE>::init(
            py::array_t<int64_t> classlabels_int64s,
            const std::vector<std::string>& classlabels_strings,
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            const std::string& post_transform,
            py::array_t<NTYPE> prob_a,
            py::array_t<NTYPE> prob_b,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors,
            py::array_t<int64_t> vectors_per_class
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
      degree_ = kernel_params_local[2];
    }
    else {
      gamma_ = 0.f;
      coef0_ = 0.f;
      degree_ = 0.f;
    }

    array2vector(proba_, prob_a, NTYPE);
    array2vector(probb_, prob_b, NTYPE);
    array2vector(vectors_per_class_, vectors_per_class, int64_t);
    if (classlabels_strings.size() > 0)
        throw std::runtime_error("This runtime only handles integers.");
    // classlabels_strings_ = classlabels_strings;
    array2vector(classlabels_ints_, classlabels_int64s, int64_t);
    
    Initialize();
}


template<typename NTYPE>
void RuntimeSVMClassifier<NTYPE>::Initialize() {
  vector_count_ = 0;
  feature_count_ = 0;
  class_count_ = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(vectors_per_class_.size()); i++) {
    starting_vector_.push_back(vector_count_);
    vector_count_ += vectors_per_class_[i];
  }

  if (classlabels_ints_.size() > 0) {
    class_count_ = classlabels_ints_.size();
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


template<typename NTYPE>
int _set_score_svm(int64_t* output_data, NTYPE max_weight, const int64_t maxclass,
                   const int64_t n, POST_EVAL_TRANSFORM post_transform_,
                   const std::vector<NTYPE>& proba_, bool weights_are_all_positive_,
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


template<typename NTYPE>
py::tuple RuntimeSVMClassifier<NTYPE>::compute(py::array_t<NTYPE> X) const {
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
        nb_columns = class_count_ > 2
                        ? nb_columns = class_count_ * (class_count_ - 1) / 2
                        : 2;
    }

    std::vector<int64_t> dims{N, nb_columns};    
                        
    py::array_t<int64_t> Y(N); // one target only
    py::array_t<NTYPE> Z(N * nb_columns); // one target only
    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Y, Z, nb_columns);
    }
    return py::make_tuple(Y, Z);
}

template<typename NTYPE>
NTYPE RuntimeSVMClassifier<NTYPE>::kernel_dot_gil_free(
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


template<typename NTYPE>
void multiclass_probability(int64_t classcount, const std::vector<NTYPE>& r,
                            std::vector<NTYPE>& p) {
  int64_t sized2 = classcount * classcount;
  std::vector<NTYPE> Q(size2, 0);
  std::vector<NTYPE> Qp(classcount, 0);
  NTYPE eps = 0.005f / static_cast<NTYPE>(classcount);
  for (int64_t i = 0; i < classcount; i++) {
    p[i] = 1.0f / static_cast<NTYPE>(classcount);  // Valid if k = 1
    for (int64_t j = 0; j < i; j++) {
      Q[i * classcount + i] += r[j * classcount + i] * r[j * classcount + i];
      Q[i * classcount + j] = Q[j * classcount + i];
    }
    for (int64_t j = i + 1; j < classcount; j++) {
      Q[i * classcount + i] += r[j * classcount + i] * r[j * classcount + i];
      Q[i * classcount + j] = -r[j * classcount + i] * r[i * classcount + j];
    }
  }
  NTYPE pQp, max_error, error, diff;
  for (int64_t loop = 0; loop < 100; loop++) {
    // stopping condition, recalculate QP,pQP for numerical accuracy
    pQp = 0;
    for (int64_t i = 0; i < classcount; i++) {
      Qp[i] = 0;
      for (int64_t j = 0; j < classcount; j++) {
        Qp[i] += Q[i * classcount + j] * p[j];
      }
      pQp += p[i] * Qp[i];
    }
    max_error = 0;
    for (int64_t i = 0; i < classcount; i++) {
      error = std::fabs(Qp[i] - pQp);
      if (error > max_error) {
        max_error = error;
      }
    }
    if (max_error < eps)
      break;

    for (int64_t i = 0; i < classcount; i++) {
      diff = (-Qp[i] + pQp) / Q[i * classcount + i];
      p[i] += diff;
      pQp = (pQp + diff * (diff * Q[i * classcount + i] + 2 * Qp[i])) / (1 + diff) / (1 + diff);
      for (int64_t j = 0; j < classcount; j++) {
        Qp[j] = (Qp[j] + diff * Q[i * classcount + j]) / (1 + diff);
        p[j] /= (1 + diff);
      }
    }
  }
}


py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(py::array_t<float>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(py::array_t<double>& Z) {
    return Z.mutable_unchecked<1>();
}


template<typename NTYPE>
void RuntimeSVMClassifier<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X,
                py::array_t<int64_t>& Y, py::array_t<NTYPE>& Z,
                int64_t nb_columns) const {
  auto Y_ = Y.mutable_unchecked<1>();
  auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
  const NTYPE* x_data = X.data(0);
  int64_t* y_data = (int64_t*)Y_.data(0);
  NTYPE* z_data = (NTYPE*)Z_.data(0);

  int64_t zindex = 0; 
  for (int64_t n = 0; n < N; ++n)  //for each example
  {
    int64_t current_weight_0 = n * stride;
    int64_t maxclass = -1;
    std::vector<NTYPE> decisions;
    std::vector<NTYPE> scores;
    std::vector<NTYPE> kernels;
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
          NTYPE sum = 0;
          int64_t start_index_i = starting_vector_[i];  // *feature_count_;
          int64_t start_index_j = starting_vector_[j];  // *feature_count_;

          int64_t class_i_support_count = vectors_per_class_[i];
          int64_t class_j_support_count = vectors_per_class_[j];

          int64_t pos1 = (vector_count_) * (j - 1);
          int64_t pos2 = (vector_count_) * (i);
          const NTYPE* val1 = &(coefficients_[pos1 + start_index_i]);
          const NTYPE* val2 = &(kernels[start_index_i]);
          for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          val1 = &(coefficients_[pos2 + start_index_j]);
          val2 = &(kernels[start_index_j]);
          for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          sum += rho_[evals];
          scores.push_back((NTYPE)sum);
          ++(votes[sum > 0 ? i : j]);
          ++evals;  //index into rho
        }
      }
    }

    if (proba_.size() > 0 && mode_ == SVM_TYPE::SVM_SVC) {
      //compute probabilities from the scores
      int64_t num = class_count_ * class_count_;
      std::vector<NTYPE> probsp2(num, 0.f);
      std::vector<NTYPE> estimates(class_count_, 0.f);
      int64_t index = 0;
      NTYPE val1, val2;
      for (int64_t i = 0; i < class_count_; ++i) {
        int64_t p1 = i * class_count_ + i + 1;
        int64_t p2 = (i + 1) * class_count_ + i;
        for (int64_t j = i + 1; j < class_count_; ++j, ++index) {
          val1 = sigmoid_probability(scores[index], proba_[index], probb_[index]);
          val2 = std::max(val1, (NTYPE)1.0e-7);
          val2 = std::min(val2, (NTYPE)(1 - 1.0e-7));
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

    NTYPE max_weight = 0;
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
          weights_are_all_positive_, classlabels_ints_, 1, 0);
    } else if (classlabels_ints_.size() > 0) {  //multiclass
        y_data[n] = classlabels_ints_[maxclass];
    } else {
        y_data[n] = maxclass;
    }

    write_scores(scores, post_transform_, z_data + zindex, write_additional_scores);
    zindex += scores.size();
  }
}

class RuntimeSVMClassifierFloat : public RuntimeSVMClassifier<float>
{
    public:
        RuntimeSVMClassifierFloat() : RuntimeSVMClassifier<float>() {}
};


class RuntimeSVMClassifierDouble : public RuntimeSVMClassifier<double>
{
    public:
        RuntimeSVMClassifierDouble() : RuntimeSVMClassifier<double>() {}
};


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

    py::class_<RuntimeSVMClassifierFloat> clf (m, "RuntimeSVMClassifierFloat",
        R"pbdoc(Implements runtime for operator SVMClassifier. The code is inspired from
`svm_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &RuntimeSVMClassifierFloat::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    clf.def("compute", &RuntimeSVMClassifierFloat::compute,
            "Computes the predictions for the SVM classifier.");
    clf.def("runtime_options", &RuntimeSVMClassifierFloat::runtime_options,
            "Returns indications about how the runtime was compiled.");
    clf.def("omp_get_max_threads", &RuntimeSVMClassifierFloat::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    py::class_<RuntimeSVMClassifierDouble> cld (m, "RuntimeSVMClassifierDouble",
        R"pbdoc(Implements runtime for operator SVMClassifierDouble. The code is inspired from
`svm_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &RuntimeSVMClassifierDouble::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cld.def("compute", &RuntimeSVMClassifierDouble::compute,
            "Computes the predictions for the SVM classifier.");
    cld.def("runtime_options", &RuntimeSVMClassifierDouble::runtime_options,
            "Returns indications about how the runtime was compiled.");
    cld.def("omp_get_max_threads", &RuntimeSVMClassifierDouble::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");
}

#endif
