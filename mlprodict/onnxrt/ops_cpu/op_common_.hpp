#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <iterator>


enum class POST_EVAL_TRANSFORM {
  NONE,
  LOGISTIC,
  SOFTMAX,
  SOFTMAX_ZERO,
  PROBIT
};

POST_EVAL_TRANSFORM to_POST_EVAL_TRANSFORM(const std::string &value);

enum class NODE_MODE {
  BRANCH_LEQ,
  BRANCH_LT,
  BRANCH_GTE,
  BRANCH_GT,
  BRANCH_EQ,
  BRANCH_NEQ,
  LEAF
};

NODE_MODE to_NODE_MODE(const std::string &value);

enum class AGGREGATE_FUNCTION {
  AVERAGE,
  SUM,
  MIN,
  MAX
};

AGGREGATE_FUNCTION to_AGGREGATE_FUNCTION(const std::string& input);

enum class SVM_TYPE {
  SVM_LINEAR,
  SVM_SVC
};

SVM_TYPE to_SVM_TYPE(const std::string &value);

enum KERNEL {
  LINEAR,
  POLY,
  RBF,
  SIGMOID
};

KERNEL to_KERNEL(const std::string &value);

static inline float ErfInv(float x) {
  float sgn = x < 0 ? -1.0f : 1.0f;
  x = (1 - x) * (1 + x);
  float log = std::log(x);
  float v = 2 / (3.14159f * 0.147f) + 0.5f * log;
  float v2 = 1 / (0.147f) * log;
  float v3 = -v + std::sqrt(v * v - v2);
  x = sgn * std::sqrt(v3);
  return x;
}


static inline float ComputeLogistic(float val) {
  float v = 1 / (1 + std::exp(-std::abs(val)));
  return (val < 0) ? (1 - v) : v;
}


static const float ml_sqrt2 = 1.41421356f;


static inline float ComputeProbit(float val) {
  return ml_sqrt2 * ErfInv(2 * val - 1);
}

static inline float sigmoid_probability(float score, float proba, float probb) {
  float val = score * proba + probb;
  return 1 - ComputeLogistic(val);  // ref: https://github.com/arnaudsj/libsvm/blob/eaaefac5ebd32d0e07902e1ae740e038eaaf0826/svm.cpp#L1818
}


void ComputeSoftmax(std::vector<float>& values);
void ComputeSoftmaxZero(std::vector<float>& values);
void write_scores(std::vector<float>& scores, POST_EVAL_TRANSFORM post_transform,
                  float* Z, int add_second_class);


#define array2vector(vec, arr, dtype) { \
    if (arr.size() > 0) { \
        auto n = arr.size(); \
        auto p = (dtype*) arr.data(0); \
        vec = std::vector<dtype>(p, p + n); \
    } \
}

#define arrayshape2vector(vec, arr) { \
    if (arr.size() > 0) { \
        vec.resize(arr.ndim()); \
        for(size_t i = 0; i < vec.size(); ++i) \
            vec[i] = (int64_t) arr.shape(i); \
    } \
}


