// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef SKIP_PYTHON
//#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include <numpy/arrayobject.h>

#if USE_OPENMP
#include <omp.h>
#endif

#include <memory>

namespace py = pybind11;
#endif

#include "op_common_.hpp"

//////////
// classes
//////////

// NgrampPart implements a Trie like structure
// for a unigram (1) it would insert into a root map with a valid id.
// for (1,2,3) node 2 would be a child of 1 but have id == 0
// because (1,2) does not exists. Node 3 would have a valid id.
template <class T>
class NgramPart;

template <>
class NgramPart<int64_t>;

using NgramPartInt = NgramPart<int64_t>;

class IntMap : public std::unordered_map<int64_t, NgramPartInt*> {
    public:
        IntMap() : std::unordered_map<int64_t, NgramPartInt*>() { }
        ~IntMap() {
            for(auto it = begin(); it != end(); ++it)
                delete it->second;
        }
};


template <>
class NgramPart<int64_t> {
    public:
        size_t id_;  // 0 - means no entry, search for a bigger N
        IntMap leafs_;
        NgramPart(size_t id) : id_(id) {}
        ~NgramPart() { }
};


// The weighting criteria.
// "TF"(term frequency),
//    the counts are propagated to output
// "IDF"(inverse document frequency),
//    all the counts larger than 1
//    would be truncated to 1 and the i-th element
//    in weights would be used to scale (by multiplication)
//    the count of the i-th n-gram in pool
// "TFIDF" (the combination of TF and IDF).
//  counts are scaled by the associated values in the weights attribute.

enum WeightingCriteria {
    kNone = 0,
    kTF = 1,
    kIDF = 2,
    kTFIDF = 3
};


class RuntimeTfIdfVectorizer {
    public:
        RuntimeTfIdfVectorizer();
    
        void Init(int max_gram_length,
                  int max_skip_count,
                  int min_gram_length,
                  const std::string& mode,
                  const std::vector<int64_t>& ngram_counts,
                  const std::vector<int64_t>& ngram_indexes,
                  const std::vector<int64_t>& pool_int64s,
                  const std::vector<float>& weights);
        ~RuntimeTfIdfVectorizer() { }

        py::array_t<float> Compute(py::array_t<int64_t> X) const;

    private:

        void ComputeImpl(const py::array_t<int64_t>& X, 
                         ptrdiff_t row_num, size_t row_size,
                         std::vector<uint32_t>& frequencies) const;

        py::array_t<float> OutputResult(size_t b_dim, const std::vector<uint32_t>& frequences) const;

    private:
    
        WeightingCriteria weighting_criteria_;
        int64_t max_gram_length_;
        int64_t min_gram_length_;
        int64_t max_skip_count_;
        std::vector<int64_t> ngram_counts_;
        std::vector<int64_t> ngram_indexes_;
        std::vector<float> weights_;
        std::vector<int64_t> pool_int64s_;
        IntMap int64_map_;
        size_t output_size_ = 0;
      
        void IncrementCount(size_t ngram_id, size_t row_num,
                            std::vector<uint32_t>& frequencies) const {
            // assert(ngram_id != 0);
            --ngram_id;
            // assert(ngram_id < ngram_indexes_.size());
            auto output_idx = row_num * output_size_ + ngram_indexes_[ngram_id];
            // assert(static_cast<size_t>(output_idx) < frequencies.size());
            ++frequencies[output_idx];
        }
};


/////////
// ngrams
/////////


// Returns next ngram_id
template <class K, class ForwardIter, class Map>
inline size_t PopulateGrams(ForwardIter first, size_t ngrams, size_t ngram_size,
                            size_t ngram_id, Map& c) {
    for (; ngrams > 0; --ngrams) {
        size_t n = 1;
        Map* m = &c;
        while (true) {
            auto p = m->emplace(*first, new NgramPart<int64_t>(0));
            ++first;
            if (n == ngram_size) {
                p.first->second->id_ = ngram_id;
                ++ngram_id;
                break;
            }
            ++n;
            m = &p.first->second->leafs_;
        }
    }
    return ngram_id;
}


////////
// tools
////////

inline const void* AdvanceElementPtr(const void* p, size_t elements, size_t element_size) {
    return reinterpret_cast<const uint8_t*>(p) + elements * element_size;
}

//////////////////
// TfIdfVectorizer
//////////////////

RuntimeTfIdfVectorizer::RuntimeTfIdfVectorizer() {
    weighting_criteria_ = WeightingCriteria::kNone;
    max_gram_length_ = 0;
    min_gram_length_ = 0;
    max_skip_count_ = 0;
    output_size_ = 0;
}

void RuntimeTfIdfVectorizer::Init(
        int max_gram_length, int max_skip_count, int min_gram_length,
        const std::string& mode, const std::vector<int64_t>& ngram_counts,
        const std::vector<int64_t>& ngram_indexes,
        const std::vector<int64_t>& pool_int64s,
        const std::vector<float>& weights) {
    if (mode == "TF")
        weighting_criteria_ = kTF;
    else if (mode == "IDF")
        weighting_criteria_ = kIDF;
    else if (mode == "TFIDF")
        weighting_criteria_ = kTFIDF;

    min_gram_length_ = min_gram_length;
    max_gram_length_ = max_gram_length;
    max_skip_count_ = max_skip_count;
    ngram_counts_ = ngram_counts;
    max_gram_length_ = max_gram_length;
    ngram_indexes_ = ngram_indexes;

    auto greatest_hit = std::max_element(ngram_indexes_.cbegin(), ngram_indexes_.cend());
    output_size_ = *greatest_hit + 1;

    weights_ = weights;
    pool_int64s_ = pool_int64s;

    const auto total_items = pool_int64s.size();
    size_t ngram_id = 1;  // start with 1, 0 - means no n-gram
    // Load into dictionary only required gram sizes
    size_t ngram_size = 1;
    for (size_t i = 0; i < ngram_counts_.size(); ++i) {
        
        size_t start_idx = ngram_counts_[i];
        size_t end_idx = ((i + 1) < ngram_counts_.size()) 
                            ? ngram_counts_[i + 1] : total_items;
        auto items = end_idx - start_idx;
        if (items > 0) {
            auto ngrams = items / ngram_size;
            if ((int)ngram_size >= min_gram_length && (int)ngram_size <= max_gram_length)
                ngram_id = PopulateGrams<int64_t>(
                    pool_int64s.begin() + start_idx, ngrams, ngram_size,
                    ngram_id, int64_map_);
            else
                ngram_id += ngrams;
        }
        ++ngram_size;
    }
}


py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(py::array_t<float>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(py::array_t<double>& Z) {
    return Z.mutable_unchecked<1>();
}


py::array_t<float> RuntimeTfIdfVectorizer::OutputResult(
        size_t B, const std::vector<uint32_t>& frequences) const {
    std::vector<int64_t> output_dims;
    if (B == 0) {
        output_dims.push_back(output_size_);
        B = 1;  // For use in the loops below
    }
    else {
        output_dims.push_back(B);
        output_dims.push_back(output_size_);
    }

    const auto row_size = output_size_;

    auto total_dims = flattened_dimension(output_dims);
    py::array_t<float> Y(total_dims);
    auto output_data_ = _mutable_unchecked1(Y);
    float* output_data = (float*)output_data_.data(0);

    const auto& w = weights_;
    switch (weighting_criteria_) {
        case kTF: {
            for (auto f : frequences)
                *output_data++ = static_cast<float>(f);
        } break;
        case kIDF: {
            if (!w.empty()) {
                const auto* freqs = frequences.data();
                for (size_t batch = 0; batch < B; ++batch)
                    for (size_t i = 0; i < row_size; ++i)
                        *output_data++ = (*freqs++ > 0) ? w[i] : 0;
            }
            else {
                for (auto f : frequences)
                    *output_data++ = (f > 0) ? 1.0f : 0;
            }
        } break;
        case kTFIDF: {
            if (!w.empty()) {
                const auto* freqs = frequences.data();
                for (size_t batch = 0; batch < B; ++batch)
                    for (size_t i = 0; i < row_size; ++i)
                        *output_data++ = *freqs++ * w[i];
            }
            else {
                for (auto f : frequences)
                    *output_data++ = static_cast<float>(f);
            }
        } break;
        case kNone:  // fall-through
        default:
            throw std::runtime_error("Unexpected weighting_criteria.");
    }
    return Y;
}

void RuntimeTfIdfVectorizer::ComputeImpl(
        const py::array_t<int64_t>& X, ptrdiff_t row_num, size_t row_size,
        std::vector<uint32_t>& frequencies) const {
    const auto elem_size = sizeof(int64_t);

    const void* row_begin = AdvanceElementPtr((void*)X.data(0), row_num * row_size, elem_size);
    const void* const row_end = AdvanceElementPtr(row_begin, row_size, elem_size);

    const auto max_gram_length = max_gram_length_;
    const auto max_skip_distance = max_skip_count_ + 1;  // Convert to distance
    auto start_ngram_size = min_gram_length_;

    for (auto skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
        auto ngram_start = row_begin;
        auto const ngram_row_end = row_end;

        while (ngram_start < ngram_row_end) {
            // We went far enough so no n-grams of any size can be gathered
            auto at_least_this = AdvanceElementPtr(
                ngram_start, skip_distance * (start_ngram_size - 1), elem_size);
            if (at_least_this >= ngram_row_end)
                break;

            auto ngram_item = ngram_start;
            const IntMap* int_map = &int64_map_;
            for (auto ngram_size = 1;
                    !int_map->empty() &&
                    ngram_size <= max_gram_length &&
                    ngram_item < ngram_row_end;
                    ++ngram_size, ngram_item = AdvanceElementPtr(ngram_item, skip_distance, elem_size)) {
                int64_t val = *reinterpret_cast<const int64_t*>(ngram_item);
                auto hit = int_map->find(val);
                if (hit == int_map->end())
                    break;
                if (ngram_size >= start_ngram_size && hit->second->id_ != 0)
                    IncrementCount(hit->second->id_, row_num, frequencies);
                int_map = &hit->second->leafs_;
            }
            // Sliding window shift
            ngram_start = AdvanceElementPtr(ngram_start, 1, elem_size);
        }
        // We count UniGrams only once since they are not affected
        // by skip distance
        if (start_ngram_size == 1 && ++start_ngram_size > max_gram_length)
            break;
    }
}

py::array_t<float> RuntimeTfIdfVectorizer::Compute(py::array_t<int64_t> X) const {
    std::vector<int64_t> input_shape;
    arrayshape2vector(input_shape, X);
    const size_t total_items = flattened_dimension(input_shape);

    int32_t num_rows = 0;
    size_t B = 0;
    size_t C = 0;
    auto& input_dims = input_shape;
    if (input_dims.empty()) {
        num_rows = 1;
        C = 1;
        if (total_items != 1)
            throw std::runtime_error("Unexpected total of items.");
    }
    else if (input_dims.size() == 1) {
        num_rows = 1;
        C = input_dims[0];
    }
    else if (input_dims.size() == 2) {
        B = input_dims[0];
        C = input_dims[1];
        num_rows = static_cast<int32_t>(B);
        if (B < 1)
            throw std::runtime_error(
                "Input shape must have either [C] or [B,C] dimensions with B > 0.");
    }
    else
        throw std::runtime_error(
                  "Input shape must have either [C] or [B,C] dimensions with B > 0.");

    if (num_rows * C != total_items)
        throw std::runtime_error("Unexpected total of items.");
    // Frequency holder allocate [B..output_size_]
    // and init all to zero
    std::vector<uint32_t> frequencies;
    frequencies.resize(num_rows * output_size_, 0);

    if (total_items == 0 || int64_map_.empty()) {
        // TfidfVectorizer may receive an empty input when it follows a Tokenizer
        // (for example for a string containing only stopwords).
        // TfidfVectorizer returns a zero tensor of shape
        // {b_dim, output_size} when b_dim is the number of received observations
        // and output_size the is the maximum value in ngram_indexes attribute plus 1.
        return OutputResult(B, frequencies);
    }

    std::function<void(ptrdiff_t)> fn = [this, X, C, &frequencies](ptrdiff_t row_num) {
        ComputeImpl(X, row_num, C, frequencies);
    };

    // can be parallelized.
    for (int64_t i = 0; i < num_rows; ++i)
        fn(i);

    return OutputResult(B, frequencies);
}


/////////
// python
/////////


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_tfidfvectorizer_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator TfIdfVectorizer."
    #else
    R"pbdoc(Implements runtime for operator TfIdfVectorizer. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/tfidfvectorizer.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeTfIdfVectorizer> cli (m, "RuntimeTfIdfVectorizer",
        R"pbdoc(Implements runtime for operator TfIdfVectorizer. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/tfidfvectorizer.cc>`_
in :epkg:`onnxruntime`. Supports Int only.)pbdoc");

    cli.def(py::init<>());
    cli.def("init", &RuntimeTfIdfVectorizer::Init, "Initializes TfIdf.");
    cli.def("compute", &RuntimeTfIdfVectorizer::Compute, "Computes TfIdf.");
}

#endif
