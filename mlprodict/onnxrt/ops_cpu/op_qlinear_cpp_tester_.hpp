#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/providers/cpu/nn/qlinearconv_op_test.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "op_conv_matrices_.hpp"
#include <numeric>
#include <random>
#include <map>

namespace detail {

    class RuntimeTesterIO {
    public:
        int type_;
        std::string name_;
        std::vector<int64_t> shape_;
        std::vector<float> values_float_;  // 1
        std::vector<int64_t> values_int64_t_;  // 2
        std::vector<uint8_t> values_uint8_t_;  // 3
        std::vector<int8_t> values_int8_t_;  // 4
        std::vector<int32_t> values_int32_t_;  // 5
    public:
        RuntimeTesterIO() : type_(0), name_(), shape_() {}
        RuntimeTesterIO(const RuntimeTesterIO& copy) : type_(copy.type_), name_(copy.name_), shape_(copy.shape_),
            values_float_(copy.values_float_), values_int64_t_(copy.values_int64_t_),
            values_uint8_t_(copy.values_uint8_t_), values_int8_t_(copy.values_int8_t_),
            values_int32_t_(copy.values_int32_t_) {}
        RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<float>& values) : name_(name), shape_(shape), values_float_(values) { type_ = 1; }
        RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<int64_t>& values) : name_(name), shape_(shape), values_int64_t_(values) { type_ = 2; }
        RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<uint8_t>& values) : name_(name), shape_(shape), values_uint8_t_(values) { type_ = 3; }
        RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<int8_t>& values) : name_(name), shape_(shape), values_int8_t_(values) { type_ = 4; }
        RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<int32_t>& values) : name_(name), shape_(shape), values_int32_t_(values) { type_ = 5; }
        RuntimeTesterIO(const char* name, const int64_t& value) : name_(name), shape_(), values_int64_t_() {
            type_ = 2;
            values_int64_t_.push_back(value);
        }

        std::string to_string(const std::string& sep = ",") const {
            std::ostringstream st;
            for (size_t i = 0; i < shape_.size(); ++i)
                st << shape_[i] << "x";
            st << ":";
            switch (type_) {
            case 1:
                st << "float:";
                for (size_t i = 0; i < values_float_.size(); ++i)
                    st << values_float_[i] << sep;
                break;
            case 2:
                st << "int64_t:";
                for (size_t i = 0; i < values_int64_t_.size(); ++i)
                    st << values_int64_t_[i] << sep;
                break;
            case 3:
                st << "uint8_t:";
                for (size_t i = 0; i < values_uint8_t_.size(); ++i)
                    st << (int)values_uint8_t_[i] << sep;
                break;
            case 4:
                st << "int8_t:";
                for (size_t i = 0; i < values_int8_t_.size(); ++i)
                    st << (int)values_int8_t_[i] << sep;
                break;
            case 5:
                st << "int32_t:";
                for (size_t i = 0; i < values_int32_t_.size(); ++i)
                    st << values_int32_t_[i] << sep;
                break;
            default:
                throw std::invalid_argument("Unexpected type.");
            }
            return st.str();
        }
    };

    template <typename T>
    inline T GetValue(const RuntimeTesterIO& io) {
        throw std::invalid_argument(MakeString("Unable to get value (type=", io.type_, ")."));
    }

    template <>
    inline int64_t GetValue<int64_t>(const RuntimeTesterIO& io) {
        if (io.type_ != 2 || io.values_int64_t_.size() != 1 || io.shape_.size() != 0)
            throw std::invalid_argument("Unexpected error.");
        return io.values_int64_t_[0];
    }

    template <typename T>
    inline std::vector<T> GetVectorValue(const RuntimeTesterIO& io) {
        throw std::invalid_argument(MakeString("Unable to get vector value (type=", io.type_, ")."));
    }

    template <>
    inline std::vector<float> GetVectorValue<float>(const RuntimeTesterIO& io) {
        if (io.type_ != 1)
            throw std::invalid_argument("Unexpected error.");
        return io.values_float_;
    }

    template <>
    inline std::vector<int64_t> GetVectorValue<int64_t>(const RuntimeTesterIO& io) {
        if (io.type_ != 2)
            throw std::invalid_argument("Unexpected error.");
        return io.values_int64_t_;
    }

    template <>
    inline  std::vector<uint8_t> GetVectorValue<uint8_t>(const RuntimeTesterIO& io) {
        if (io.type_ != 3)
            throw std::invalid_argument("Unexpected error.");
        return io.values_uint8_t_;
    }

    template <>
    inline std::vector<int8_t> GetVectorValue<int8_t>(const RuntimeTesterIO& io) {
        if (io.type_ != 4)
            throw std::invalid_argument("Unexpected error.");
        return io.values_int8_t_;
    }

    template <>
    inline std::vector<int32_t> GetVectorValue<int32_t>(const RuntimeTesterIO& io) {
        if (io.type_ != 5)
            throw std::invalid_argument("Unexpected error.");
        return io.values_int32_t_;
    }
}

class RuntimeTesterIO : public detail::RuntimeTesterIO {
public:
    RuntimeTesterIO() : detail::RuntimeTesterIO() {}
    RuntimeTesterIO(const RuntimeTesterIO& copy) : detail::RuntimeTesterIO(copy) {}
    RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<float>& values) : detail::RuntimeTesterIO(name, shape, values) {}
    RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<int64_t>& values) : detail::RuntimeTesterIO(name, shape, values) {}
    RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<uint8_t>& values) : detail::RuntimeTesterIO(name, shape, values) {}
    RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<int8_t>& values) : detail::RuntimeTesterIO(name, shape, values) {}
    RuntimeTesterIO(const char* name, const std::vector<int64_t>& shape, const std::vector<int32_t>& values) : detail::RuntimeTesterIO(name, shape, values) {}
    RuntimeTesterIO(const char* name, const int64_t& value) : detail::RuntimeTesterIO(name, value) {}

    template <typename T>
    T GetValue() const { return detail::GetValue<T>(*this); }

    template <typename T>
    std::vector<T> GetVectorValue() const { return detail::GetVectorValue<T>(*this); }

    template <typename T>
    shaped_array_t<T> GetArrayValue() const { return shaped_array_t<T>(GetVectorValue<T>(), shape_); }
};


class RuntimeTesterO : public RuntimeTesterIO {
public:
    bool check_;
    float error_;
public:
    RuntimeTesterO() : RuntimeTesterIO(), check_(false), error_(0) {}
    RuntimeTesterO(const RuntimeTesterO& copy) : RuntimeTesterIO(copy), check_(copy.check_), error_(copy.error_) {}
    RuntimeTesterO(const char* name, const std::vector<int64_t>& shape, const std::vector<float>& values, bool check = true, float error = 0) : RuntimeTesterIO(name, shape, values), check_(check), error_(error) {}
    RuntimeTesterO(const char* name, const std::vector<int64_t>& shape, const std::vector<int64_t>& values, bool check = true, float error = 0) : RuntimeTesterIO(name, shape, values), check_(check), error_(error) {}
    RuntimeTesterO(const char* name, const std::vector<int64_t>& shape, const std::vector<uint8_t>& values, bool check = true, float error = 0) : RuntimeTesterIO(name, shape, values), check_(check), error_(error) {}
    RuntimeTesterO(const char* name, const std::vector<int64_t>& shape, const std::vector<int8_t>& values, bool check = true, float error = 0) : RuntimeTesterIO(name, shape, values), check_(check), error_(error) {}
    RuntimeTesterO(const char* name, const std::vector<int64_t>& shape, const std::vector<int32_t>& values, bool check = true, float error = 0) : RuntimeTesterIO(name, shape, values), check_(check), error_(error) {}

    template <typename T>
    bool Check(const shaped_array_t<T>& got) {
        if (!check_)
            return true;
        return GetArrayValue<T>().equal(got);
    }
};


class RuntimeTester {
protected:
    int opset_;
    std::string op_name_;
    std::vector<RuntimeTesterIO> inputs_;
    std::vector<RuntimeTesterO> outputs_;
    std::map<std::string, RuntimeTesterIO> attributes_;
public:
    RuntimeTester(const char* op_name, int opset = 13) : op_name_(op_name) {
        opset_ = opset;
    }

    template<typename T>
    void AddInput(const char* name, const std::vector<int64_t>& shape, const std::vector<T>& input) {
        RuntimeTesterIO io(name, shape, input);
        inputs_.push_back(io);
    }

    template<typename T>
    void AddOutput(
        const char* name, const std::vector<int64_t>& shape, const std::vector<T>& output,
        bool check_output = true, float error = 0) {
        RuntimeTesterO io(name, shape, output, check_output, error);
        outputs_.push_back(io);
    }

    template<typename T>
    void AddAttribute(const char* name, const std::vector<T>& output) {
        std::vector<int64_t> shape{ (int64_t)output.size() };
        RuntimeTesterIO at(name, shape, output);
        attributes_[name] = at;
    }
    template<typename T>
    void AddAttribute(const char* name, const T& output) {
        RuntimeTesterIO at(name, output);
        attributes_[name] = at;
    }

    template <typename T>
    T GetAttribute(const std::string& name, const T& default_value) {
        auto it = attributes_.find(name);
        if (it == attributes_.end())
            return default_value;
        return it->second.GetValue<T>();
    }

    std::string GetAttributeString(const std::string& name, const std::string& default_value) {
        auto it = attributes_.find(name);
        if (it == attributes_.end())
            return default_value;
        return it->second.GetValue<std::string>();
    }

    template <typename T>
    std::vector<T> GetVectorAttribute(const std::string& name) {
        auto it = attributes_.find(name);
        if (it == attributes_.end())
            return std::vector<T>();
        return it->second.GetVectorValue<T>();
    }

    template <typename T>
    shaped_array_t<T> GetArrayAttribute(const std::string& name) {
        auto it = attributes_.find(name);
        if (it == attributes_.end())
            return shaped_array_t<T>();
        std::vector<T> value = it->second.GetVectorValue<T>();
        return shaped_array_t<T>(value, { (int64_t)value.size() });
    }

    template <typename T>
    void CheckSameType(const std::vector<shaped_array_t<T>>& res) {
        if (res.size() != outputs_.size())
            throw std::invalid_argument(MakeString(res.size(), " outputs but ", outputs_.size(), " are expected."));
        for (size_t i = 0; i < res.size(); ++i) {
            if (!outputs_[i].Check(res[0]))
                throw std::invalid_argument(MakeString("output ", i, " is different."));
        }
    }

    virtual void Run(bool expect_success, const char* ignored = NULL) {
        throw std::invalid_argument(MakeString("Not implemented for ',", op_name_, "'."));
    }
};
