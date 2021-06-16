
#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "op_common_.hpp"
#include <string.h> // memcpy
#include <stdlib.h> // realloc


POST_EVAL_TRANSFORM to_POST_EVAL_TRANSFORM(const std::string& value) {
    if (value.compare("NONE") == 0) return POST_EVAL_TRANSFORM::NONE;
    if (value.compare("LOGISTIC") == 0) return POST_EVAL_TRANSFORM::LOGISTIC;
    if (value.compare("SOFTMAX") == 0) return POST_EVAL_TRANSFORM::SOFTMAX;
    if (value.compare("SOFTMAX_ZERO") == 0) return POST_EVAL_TRANSFORM::SOFTMAX_ZERO;
    if (value.compare("PROBIT") == 0) return POST_EVAL_TRANSFORM::PROBIT;
    throw std::invalid_argument(std::string("NODE_MODE '") +
        value +
        std::string("' is not defined."));
}


NODE_MODE to_NODE_MODE(const std::string& value) {
    if (value.compare("BRANCH_LEQ") == 0) return NODE_MODE::BRANCH_LEQ;
    if (value.compare("BRANCH_LT") == 0) return NODE_MODE::BRANCH_LT;
    if (value.compare("BRANCH_GTE") == 0) return NODE_MODE::BRANCH_GTE;
    if (value.compare("BRANCH_GT") == 0) return NODE_MODE::BRANCH_GT;
    if (value.compare("BRANCH_EQ") == 0) return NODE_MODE::BRANCH_EQ;
    if (value.compare("BRANCH_NEQ") == 0) return NODE_MODE::BRANCH_NEQ;
    if (value.compare("LEAF") == 0) return NODE_MODE::LEAF;
    throw std::invalid_argument(std::string("NODE_MODE '") +
        value +
        std::string("' is not defined."));
}


const char* to_str(NODE_MODE mode) {
    switch (mode) {
    case NODE_MODE::BRANCH_LEQ: return "BRANCH_LEQ";
    case NODE_MODE::BRANCH_LT: return "BRANCH_LT";
    case NODE_MODE::BRANCH_GTE: return "BRANCH_GTE";
    case NODE_MODE::BRANCH_GT: return "BRANCH_GT";
    case NODE_MODE::BRANCH_EQ: return "BRANCH_EQ";
    case NODE_MODE::BRANCH_NEQ: return "BRANCH_NEQ";
    case NODE_MODE::LEAF: return "LEAF";
    default: return "?";
    }
}


SVM_TYPE to_SVM_TYPE(const std::string& value) {
    if (value.compare("SVM_LINEAR") == 0) return SVM_TYPE::SVM_LINEAR;
    if (value.compare("SVM_SVC") == 0) return SVM_TYPE::SVM_SVC;
    throw std::invalid_argument(std::string("SVM_TYPE '") +
        value +
        std::string("' is not defined."));
}

KERNEL to_KERNEL(const std::string& value) {
    if (value.compare("LINEAR") == 0) return KERNEL::LINEAR;
    if (value.compare("POLY") == 0) return KERNEL::POLY;
    if (value.compare("RBF") == 0) return KERNEL::RBF;
    if (value.compare("SIGMOID") == 0) return KERNEL::SIGMOID;
    throw std::invalid_argument(std::string("KERNEL '") +
        value +
        std::string("' is not defined."));
}


AGGREGATE_FUNCTION to_AGGREGATE_FUNCTION(const std::string& input) {
    if (input == "AVERAGE") return AGGREGATE_FUNCTION::AVERAGE;
    if (input == "SUM") return AGGREGATE_FUNCTION::SUM;
    if (input == "MIN") return AGGREGATE_FUNCTION::MIN;
    if (input == "MAX") return AGGREGATE_FUNCTION::MAX;
    throw std::invalid_argument(std::string("AGGREGATE_FUNCTION '") +
        input +
        std::string("' is not defined."));
}


StorageOrder to_StorageOrder(const std::string& input) {
    if (input == "UNKNOWN") return StorageOrder::UNKNOWN;
    if (input == "NHWC") return StorageOrder::NHWC;
    if (input == "NCHW") return StorageOrder::NCHW;
    throw std::invalid_argument(std::string("StorageOrder '") +
        input +
        std::string("' is not defined."));
}


AutoPadType to_AutoPadType(const std::string& input) {
    if (input == "NOTSET") return AutoPadType::NOTSET;
    if (input == "VALID") return AutoPadType::VALID;
    if (input == "SAME_UPPER") return AutoPadType::SAME_UPPER;
    if (input == "SAME_LOWER") return AutoPadType::SAME_LOWER;
    throw std::invalid_argument(std::string("AutoPadType '") +
        input +
        std::string("' is not defined."));
}


void debug_print(const std::string& msg, int64_t iter, int64_t end) {
    std::cout << msg.c_str() << ":" << iter << "/" << end << "\n";
}


void debug_print(const std::string& msg, size_t i, size_t j, size_t k, float pa, float pb, float val) {
    std::cout << msg.c_str() << ": (" << i << "," << j << "," << k << "): " << pa << "," << pb << " -> " << val << "\n";
}


void debug_print(const std::string& msg, size_t i, size_t j, size_t k, double pa, double pb, double val) {
    std::cout << msg.c_str() << ": (" << i << "," << j << "," << k << "): " << pa << "," << pb << " -> " << val << "\n";
}


template <typename T>
void debug_print_(const std::string& msg, T value) {
    std::cout << msg.c_str() << ":" << value << "\n";
}

void debug_print(const std::string& msg, float value) {
    debug_print_(msg.c_str(), value);
}


void debug_print(const std::string& msg, double value) {
    debug_print_(msg.c_str(), value);
}


void debug_print(const std::string& msg, int64_t value) {
    debug_print_(msg.c_str(), value);
}


void debug_print(const std::string& msg, size_t value) {
    debug_print_(msg.c_str(), value);
}
