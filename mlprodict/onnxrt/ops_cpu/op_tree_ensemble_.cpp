#pragma once

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "op_tree_ensemble_.hpp"


POST_EVAL_TRANSFORM to_POST_EVAL_TRANSFORM(const std::string &value)
{
    if (value.compare("NONE") == 0) return POST_EVAL_TRANSFORM::NONE;
    if (value.compare("LOGISTIC") == 0) return POST_EVAL_TRANSFORM::LOGISTIC;
    if (value.compare("SOFTMAX") == 0) return POST_EVAL_TRANSFORM::SOFTMAX;
    if (value.compare("SOFTMAX_ZERO") == 0) return POST_EVAL_TRANSFORM::SOFTMAX_ZERO;
    if (value.compare("PROBIT") == 0) return POST_EVAL_TRANSFORM::PROBIT;
    throw std::runtime_error(std::string("NODE_MODE '") + 
                             value + 
                             std::string(" is not defined."));
}


NODE_MODE to_NODE_MODE(const std::string &value)
{
    if (value.compare("BRANCH_LEQ") == 0) return NODE_MODE::BRANCH_LEQ;
    if (value.compare("BRANCH_LT") == 0) return NODE_MODE::BRANCH_LT;
    if (value.compare("BRANCH_GTE") == 0) return NODE_MODE::BRANCH_GTE;
    if (value.compare("BRANCH_GT") == 0) return NODE_MODE::BRANCH_GT;
    if (value.compare("BRANCH_EQ") == 0) return NODE_MODE::BRANCH_EQ;
    if (value.compare("BRANCH_NEQ") == 0) return NODE_MODE::BRANCH_NEQ;
    if (value.compare("LEAF") == 0) return NODE_MODE::LEAF;
    throw std::runtime_error(std::string("NODE_MODE '") + 
                             value + 
                             std::string(" is not defined."));
}


AGGREGATE_FUNCTION to_AGGREGATE_FUNCTION(const std::string& input) {
    if (input == "AVERAGE") return AGGREGATE_FUNCTION::AVERAGE;
    if (input == "SUM") return AGGREGATE_FUNCTION::SUM;
    if (input == "MIN") return AGGREGATE_FUNCTION::MIN;
    if (input == "MAX") return AGGREGATE_FUNCTION::MAX;
    throw std::runtime_error(std::string("AGGREGATE_FUNCTION '") + 
                             input + 
                             std::string(" is not defined."));
}


void ComputeSoftmax(std::vector<float>& values) {
  std::vector<float> newscores;
  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (float value : values) {
    if (value > v_max)
      v_max = value;
  }
  float this_sum = 0.f;
  for (float value : values) {
    float val2 = std::exp(value - v_max);
    this_sum += val2;
    newscores.push_back(val2);
  }
  for (int64_t k = 0; k < static_cast<int64_t>(values.size()); k++) {
    values[k] = newscores[k] / this_sum;
  }
}


void ComputeSoftmaxZero(std::vector<float>& values) {
  //this function skips zero values (since exp(0) is non zero)
  std::vector<float> newscores;
  // compute exp with negative number to be numerically stable
  float v_max = -std::numeric_limits<float>::max();
  for (float value : values) {
    if (value > v_max)
      v_max = value;
  }
  float exp_neg_v_max = std::exp(-v_max);
  float this_sum = 0.f;
  for (float value : values) {
    if (value > 0.0000001f || value < -0.0000001f) {
      float val2 = std::exp(value - v_max);
      this_sum += val2;
      newscores.push_back(val2);
    } else {
      newscores.push_back(value * exp_neg_v_max);
    }
  }
  for (int64_t k = 0; k < static_cast<int64_t>(values.size()); k++) {
    values[k] = newscores[k] / this_sum;
  }
}


void write_scores(std::vector<float>& scores, POST_EVAL_TRANSFORM post_transform,
                  float* Z, int add_second_class) {
  if (scores.size() >= 2) {
    switch (post_transform) {
      case POST_EVAL_TRANSFORM::PROBIT:
        for (float& score : scores)
          score = ComputeProbit(score);
        break;
      case POST_EVAL_TRANSFORM::LOGISTIC:
        for (float& score : scores)
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
  } else if (scores.size() == 1) {  //binary case
    if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
      scores[0] = ComputeProbit(scores[0]);
    } else {
      switch (add_second_class) {
        case 0:  //0=all positive weights, winning class is positive
          scores.push_back(scores[0]);
          scores[0] = 1.f - scores[0];  //put opposite score in positive slot
          break;
        case 1:  //1 = all positive weights, winning class is negative
          scores.push_back(scores[0]);
          scores[0] = 1.f - scores[0];  //put opposite score in positive slot
          break;
        case 2:  //2 = mixed weights, winning class is positive
          if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
            scores.push_back(ComputeLogistic(scores[0]));  //ml_logit(scores[k]);
            scores[0] = ComputeLogistic(-scores[0]);
          } else {
            scores.push_back(scores[0]);
            scores[0] = -scores[0];
          }
          break;
        case 3:  //3 = mixed weights, winning class is negative
          if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
            scores.push_back(ComputeLogistic(scores[0]));  //ml_logit(scores[k]);
            scores[0] = ComputeLogistic(-scores[0]);
          } else {
            scores.push_back(-scores[0]);
          }
          break;
      }
    }
  }
  memcpy(Z, scores.data(), scores.size() * sizeof(float));
}
