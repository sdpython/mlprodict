#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <iterator>
#include <math.h>

#if defined(_WIN32) || defined(WIN32)

inline bool _isnan_(float x) { return _isnanf(x); }
inline bool _isnan_(double x) { return _isnan(x); }

#else

#include <math.h>
inline bool _isnan_(float x) { return ::isnanf(x); }
inline bool _isnan_(double x) { return ::isnan(x); }

#endif


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

const char * to_str(NODE_MODE mode);


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


enum StorageOrder {
  UNKNOWN = 0,
  NHWC = 1,
  NCHW = 2,
};

StorageOrder to_StorageOrder(const std::string &value);


enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

AutoPadType to_AutoPadType(const std::string &value);



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


static inline double ErfInv(double x) {
  double sgn = x < 0 ? -1.0 : 1.0;
  x = (1 - x) * (1 + x);
  double log = std::log(x);
  double v = 2. / (3.14159f * 0.147f) + 0.5f * log;
  double v2 = 1. / (0.147f) * log;
  double v3 = std::sqrt(v * v - v2) - v;
  return sgn * std::sqrt(v3);
}


static inline float ComputeLogistic(float val) {
  float v = 1 / (1 + std::exp(-std::abs(val)));
  return (val < 0) ? (1 - v) : v;
}


static inline double ComputeLogistic(double val) {
  double v = 1. / (1. + std::exp(-std::abs(val)));
  return (val < 0) ? (1. - v) : v;
}


static const float ml_sqrt2 = 1.41421356f;


template<class NTYPE>
static inline NTYPE ComputeProbit(NTYPE val) {
  return ml_sqrt2 * ErfInv(val * 2 - 1);
}


template<class NTYPE>
static inline NTYPE sigmoid_probability(NTYPE score, NTYPE proba, NTYPE probb) {
  NTYPE val = score * proba + probb;
  return 1 - ComputeLogistic(val);  // ref: https://github.com/arnaudsj/libsvm/blob/eaaefac5ebd32d0e07902e1ae740e038eaaf0826/svm.cpp#L1818
}


template<class NTYPE>
void ComputeSoftmax(std::vector<NTYPE>& values) {
  NTYPE v_max = -std::numeric_limits<NTYPE>::max();
  for (NTYPE value : values) {
    if (value > v_max)
      v_max = value;
  }
  NTYPE this_sum = (NTYPE)0.;
  for (NTYPE& value : values) {
    value = std::exp(value - v_max);
    this_sum += value;
  }
  for (NTYPE& value : values) 
    value /= this_sum;
}


template<class NTYPE>
void ComputeSoftmaxZero(std::vector<NTYPE>& values) {
  NTYPE v_max = -std::numeric_limits<NTYPE>::max();
  for (NTYPE value : values) {
    if (value > v_max)
      v_max = value;
  }
  NTYPE exp_neg_v_max = std::exp(-v_max);
  NTYPE this_sum = (NTYPE)0;
  for (NTYPE& value : values) {
    if (value > 0.0000001f || value < -0.0000001f) {
      value = std::exp(value - v_max);
      this_sum += value;
    } else {
      value *= exp_neg_v_max;
    }
  }
  for (NTYPE& value : values)
    value /= this_sum;
}


template<class NTYPE>
void write_scores(std::vector<NTYPE>& scores, POST_EVAL_TRANSFORM post_transform,
                  NTYPE* Z, int add_second_class) {
    if (scores.size() >= 2) {
        switch (post_transform) {
            case POST_EVAL_TRANSFORM::PROBIT:
                for (NTYPE& score : scores)
                    score = ComputeProbit(score);
                break;
            case POST_EVAL_TRANSFORM::LOGISTIC:
                for (NTYPE& score : scores)
                    score = ComputeLogistic(score);
                break;
            case POST_EVAL_TRANSFORM::SOFTMAX:
                ComputeSoftmax(scores);
                break;
            case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
                ComputeSoftmaxZero(scores);
                break;
            default:
            case POST_EVAL_TRANSFORM::NONE:
                break;
        }
        memcpy(Z, scores.data(), scores.size() * sizeof(NTYPE));
    }
    else if (scores.size() == 1) {  //binary case
        if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
            scores[0] = ComputeProbit(scores[0]);
            *Z = scores[0];
        }
        else {
            switch (add_second_class) {
                case 0:  //0=all positive weights, winning class is positive
                    scores.push_back(scores[0]);
                    scores[0] = 1.f - scores[0];  //put opposite score in positive slot
                    break;
                case 1:  //1 = all positive weights, winning class is negative
                    scores.push_back(scores[0]);
                    scores[0] = 1.f - scores[0];  //put opposite score in positive slot
                    break;
                case 2:
                case 3:  //2 = mixed weights, winning class is positive
                    if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
                        scores.push_back(ComputeLogistic(scores[0]));  //ml_logit(scores[k]);
                        scores[0] = ComputeLogistic(-scores[0]);
                    }
                    else {
                        scores.push_back(scores[0]);
                        scores[0] = -scores[0];
                    }
                    break;
                    /*
                    case 3:  //3 = mixed weights, winning class is negative
                      if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
                        scores.push_back(ComputeLogistic(scores[0]));  //ml_logit(scores[k]);
                        scores[0] = ComputeLogistic(-scores[0]);
                      } else {
                        scores.push_back(-scores[0]);
                      }
                      break;
                    */
            }
            *Z = scores[0];
            *(Z+1) = scores[1];
        }
    }
}


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


template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values)
{
    NTYPE r = 1;
    for(auto it = values.begin(); it != values.end(); ++it)
        r *= *it;
    return r;
}


template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values, int64_t first)
{
    NTYPE r = 1;
    auto end = values.begin() + first;
    for(auto it = values.begin(); it != end; ++it)
        r *= *it;
    return r;
}


template<class DIMTYPE, class NTYPE>
void shape2strides(const std::vector<DIMTYPE>& shape, 
                   std::vector<DIMTYPE>& strides, NTYPE cst)
{
    strides.resize(shape.size());
    strides[strides.size()-1] = sizeof(NTYPE);
    for(ssize_t i = strides.size()-2; i >= 0; --i)
        strides[i] = strides[i+1] * shape[i+1];
}


template <typename T, T b>
constexpr T roundUpPow2(T a) {
  return (a + (b - 1)) & (~(b - 1));
}
