// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <thread>
#include <iterator>
#include <algorithm>

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


struct TreeNodeElementId {
    int tree_id;
    int node_id;
    inline bool operator == (const TreeNodeElementId& xyz) const {
        return (tree_id == xyz.tree_id) && (node_id == xyz.node_id);
    }
    inline bool operator < (const TreeNodeElementId& xyz) const {
        return ((tree_id < xyz.tree_id) || (
                tree_id == xyz.tree_id && node_id < xyz.node_id));
    }
};


template<typename NTYPE>
struct SparseValue {
    int64_t i;
    NTYPE value;
};


enum MissingTrack {
    NONE,
    TRUE,
    FALSE
};


template<typename NTYPE>
struct TreeNodeElement {
    TreeNodeElementId id;
    int feature_id;
    NTYPE value;
    NTYPE hitrates;
    NODE_MODE mode;
    TreeNodeElement *truenode;
    TreeNodeElement *falsenode;
    MissingTrack missing_tracks;
    std::vector<SparseValue<NTYPE>> weights;
    
    bool is_not_leave;
    bool is_missing_track_true;
};


template<typename NTYPE>
class _Aggregator
{
    protected:

        size_t n_trees_;
        int64_t n_targets_or_classes_;
        POST_EVAL_TRANSFORM post_transform_;
        const std::vector<NTYPE> * base_values_;
        NTYPE origin_;
        bool use_base_values_;

    public:

        inline _Aggregator(size_t n_trees,
                           const int64_t& n_targets_or_classes,
                           POST_EVAL_TRANSFORM post_transform,
                           const std::vector<NTYPE> * base_values) : 
                n_trees_(n_trees), n_targets_or_classes_(n_targets_or_classes),
                post_transform_(post_transform), base_values_(base_values) {
            origin_ = base_values_->size() == 1 ? (*base_values_)[0] : 0.f;
            use_base_values_ = base_values_->size() == (size_t)n_targets_or_classes_;
        }

        inline void init_score(NTYPE &score, unsigned char& has_score) const { 
            score = (NTYPE)0;
            has_score = 0;
        }

        inline void ProcessTreeNodePrediction1(NTYPE* predictions, TreeNodeElement<NTYPE> * root,
                                               unsigned char* has_predictions) const {}

        void ProcessTreeNodePrediction(NTYPE* predictions, TreeNodeElement<NTYPE> * root,
                                       unsigned char* has_predictions) const {}

        void MergePrediction(int64_t n,
                             NTYPE* predictions, unsigned char* has_predictions,
                             NTYPE* predictions2, unsigned char* has_predictions2) const {}

        inline void FinalizeScores1(NTYPE* Z, NTYPE& val,
                                    unsigned char& has_scores,
                                    int64_t * Y = 0) const {
            val = has_scores ? (val + origin_) : origin_;
            *Z = post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? ComputeProbit(val) : val;
        }

        void FinalizeScores(std::vector<NTYPE>& scores,
                            std::vector<unsigned char>& has_scores,
                            NTYPE* Z, int add_second_class,
                            int64_t * Y = 0) const {
            NTYPE val;
            for (int64_t jt = 0; jt < n_targets_or_classes_; ++jt) {
                val = use_base_values_ ? (*base_values_)[jt] : 0.f;
                val += has_scores[jt] ? scores[jt] : 0;
                scores[jt] = val;
            }
            write_scores(scores, post_transform_, Z, add_second_class);
        }
};


template<typename NTYPE>
class _AggregatorSum : public _Aggregator<NTYPE>
{
    public:

        inline _AggregatorSum<NTYPE>(size_t n_trees,
                                     const int64_t& n_targets_or_classes,
                                     POST_EVAL_TRANSFORM post_transform,
                                     const std::vector<NTYPE> * base_values) :
            _Aggregator<NTYPE>(n_trees, n_targets_or_classes,
                               post_transform, base_values) { }

        inline void init_score(NTYPE &score, unsigned char& has_score) const { 
            score = (NTYPE)0;
            has_score = this->n_trees_ > 0 ? 1 : 0;
        }

        inline void ProcessTreeNodePrediction1(NTYPE* predictions,
                                               TreeNodeElement<NTYPE> * root,
                                               unsigned char* has_predictions) const {
            *predictions += root->weights[0].value;
        }

        void ProcessTreeNodePrediction(NTYPE* predictions, TreeNodeElement<NTYPE> * root,
                                       unsigned char* has_predictions) const {
            for(auto it = root->weights.begin(); it != root->weights.end(); ++it) {
                predictions[it->i] += it->value;
                has_predictions[it->i] = 1;
            }
        }

        void MergePrediction(int64_t n, NTYPE* predictions, unsigned char* has_predictions,
                             const NTYPE* predictions2, const unsigned char* has_predictions2) const {
            for(int64_t i = 0; i < n; ++i) {
                if (has_predictions2[i]) {
                    predictions[i] += predictions2[i];
                    has_predictions[i] = 1;
                }
            }
        }

        inline void MergeOnePrediction(NTYPE* predictions, unsigned char* has_predictions,
                                       const NTYPE* predictions2, const unsigned char* has_predictions2) const {
            *predictions += *has_predictions2 ? *predictions2 : 0;
            *has_predictions = *has_predictions2 ? 1 : *has_predictions;
        }
};


template<typename NTYPE>
class _AggregatorClassifier : public _AggregatorSum<NTYPE>
{
    private:

        const std::vector<int64_t> * class_labels_;
        bool binary_case_;
        bool weights_are_all_positive_;
        int64_t positive_label_;
        int64_t negative_label_;

    public:

        inline _AggregatorClassifier(size_t n_trees,
                                     const int64_t& n_targets_or_classes,
                                     POST_EVAL_TRANSFORM post_transform,
                                     const std::vector<NTYPE> * base_values,
                                     const std::vector<int64_t> * class_labels,
                                     bool binary_case,
                                     bool weights_are_all_positive,
                                     int64_t positive_label = 1,
                                     int64_t negative_label = 0) :
            _AggregatorSum<NTYPE>(n_trees, n_targets_or_classes,
                                  post_transform, base_values),
            class_labels_(class_labels), binary_case_(binary_case),
            weights_are_all_positive_(weights_are_all_positive),
            positive_label_(positive_label), negative_label_(negative_label) { }
            
        void get_max_weight(const std::vector<NTYPE>& classes, 
                            const std::vector<unsigned char>& has_scores, 
                            int64_t& maxclass, NTYPE& maxweight) const {
            maxclass = -1;
            maxweight = (NTYPE)0;
            typename std::vector<NTYPE>::const_iterator it;
            typename std::vector<unsigned char>::const_iterator itb;
            for (it = classes.begin(), itb = has_scores.begin();
               it != classes.end(); ++it, ++itb) {
                if (*itb && (maxclass == -1 || *it > maxweight)) {
                    maxclass = (int64_t)(it - classes.begin());
                    maxweight = *it;
                }
            }
        }

        int64_t _set_score_binary(int& write_additional_scores,
                                  const NTYPE* classes,
                                  const unsigned char* has_scores) const {
            NTYPE pos_weight = has_scores[1]
                                ? classes[1]
                                : (has_scores[0] ? classes[0] : (NTYPE)0);  // only 1 class
            if (binary_case_) {
                if (weights_are_all_positive_) {
                    if (pos_weight > 0.5) {
                        write_additional_scores = 0;
                        return (*class_labels_)[1];  // positive label
                    } 
                    else {
                        write_additional_scores = 1;
                        return (*class_labels_)[0];  // negative label
                    }
                }
                else {
                    if (pos_weight > 0) {
                        write_additional_scores = 2;
                        return (*class_labels_)[1];  // positive label
                    }
                    else {
                        write_additional_scores = 3;
                        return (*class_labels_)[0];  // negative label
                    }
                }
            }
            return (pos_weight > 0) 
                        ? positive_label_   // positive label
                        : negative_label_;  // negative label
        }

        inline void FinalizeScores1(NTYPE* Z, NTYPE& val,
                                    unsigned char& has_score,
                                    int64_t * Y = 0) const {
            std::vector<NTYPE> scores(2);
            unsigned char has_scores[2] = {1, 0};

            int write_additional_scores = -1;
            if (this->base_values_->size() == 2) {
                // add base values
                scores[1] = (*(this->base_values_))[1] + val;
                scores[0] = -scores[1];
                //has_score = true;
                has_scores[1] = 1;
            }
            else if (this->base_values_->size() == 1) {
                // ONNX is vague about two classes and only one base_values.
                scores[0] = val + (*(this->base_values_))[0];
                //if (!has_scores[1])
                //scores.pop_back();
                scores[0] = val;
            }
            else if (this->base_values_->size() == 0) {
                //if (!has_score)
                //  scores.pop_back();
                scores[0] = val;
            }

            *Y = _set_score_binary(write_additional_scores, &(scores[0]), has_scores);
            write_scores(scores, this->post_transform_, Z, write_additional_scores);            
        }

        void FinalizeScores(std::vector<NTYPE>& scores,
                            std::vector<unsigned char>& has_scores,
                            NTYPE* Z, int add_second_class,
                            int64_t * Y = 0) const {
            NTYPE maxweight = (NTYPE)0;
            int64_t maxclass = -1;

            int write_additional_scores = -1;
            if (this->n_targets_or_classes_ > 2) {
                // add base values
                for (int64_t k = 0, end = static_cast<int64_t>(this->base_values_->size()); k < end; ++k) {
                    if (!has_scores[k]) {
                      has_scores[k] = true;
                      scores[k] = (*(this->base_values_))[k];
                    }
                    else {
                        scores[k] += (*(this->base_values_))[k];
                    }
                }
                get_max_weight(scores, has_scores, maxclass, maxweight);
                *Y = (*class_labels_)[maxclass];
            }
            else { // binary case
                if (this->base_values_->size() == 2) {
                    // add base values
                    if (has_scores[1]) {
                        // base_value_[0] is not used.
                        // It assumes base_value[0] == base_value[1] in this case.
                        // The specification does not forbid it but does not
                        // say what the output should be in that case.
                        scores[1] = (*(this->base_values_))[1] + scores[0];
                        scores[0] = -scores[1];
                        has_scores[1] = true;
                    }
                    else {
                        // binary as multiclass
                        scores[1] += (*(this->base_values_))[1];
                        scores[0] += (*(this->base_values_))[0];
                    }
                }
                else if (this->base_values_->size() == 1) {
                    // ONNX is vague about two classes and only one base_values.
                    scores[0] += (*(this->base_values_))[0];
                    if (!has_scores[1])
                      scores.pop_back();
                }
                else if (this->base_values_->size() == 0) {
                    if (!has_scores[1])
                      scores.pop_back();
                }

                *Y = _set_score_binary(write_additional_scores, &(scores[0]), &(has_scores[0]));
            }

            write_scores(scores, this->post_transform_, Z, write_additional_scores);
        }
};


template<typename NTYPE>
class _AggregatorAverage : public _AggregatorSum<NTYPE>
{
    public:
        inline _AggregatorAverage<NTYPE>(size_t n_trees,
                                     const int64_t& n_targets_or_classes,
                                     POST_EVAL_TRANSFORM post_transform,
                                     const std::vector<NTYPE> * base_values) :
            _AggregatorSum<NTYPE>(n_trees, n_targets_or_classes,
                                  post_transform, base_values) { }

        inline void FinalizeScores1(NTYPE* Z, NTYPE& val,
                                    unsigned char& has_scores,
                                    int64_t * Y = 0) const {
            val = has_scores
                  ? val / this->n_trees_ + this->origin_
                  : this->origin_;
            *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? ComputeProbit(val) : val;
        }

        inline void FinalizeScores(std::vector<NTYPE>& scores,
                                   std::vector<unsigned char>& has_scores,
                                   NTYPE* Z,
                                   int add_second_class,
                                   int64_t * Y = 0) const {
            NTYPE val;
            for (int64_t jt = 0; jt < this->n_targets_or_classes_; ++jt) {
                val = this->use_base_values_ ? (*(this->base_values_))[jt] : 0.f;
                val += (has_scores[jt]) ? (scores[jt] / this->n_trees_) : 0;
                scores[jt] = val;
            }
            write_scores(scores, this->post_transform_, Z, add_second_class);
        }
};


template<typename NTYPE>
class _AggregatorMin : public _Aggregator<NTYPE>
{
    public:
        inline _AggregatorMin<NTYPE>(size_t n_trees,
                                     const int64_t& n_targets_or_classes,
                                     POST_EVAL_TRANSFORM post_transform,
                                     const std::vector<NTYPE> * base_values) :
            _Aggregator<NTYPE>(n_trees, n_targets_or_classes,
                               post_transform, base_values) { }

        inline void ProcessTreeNodePrediction1(NTYPE* predictions, TreeNodeElement<NTYPE> * root,
                                               unsigned char* has_predictions) const {
            *predictions = (!(*has_predictions) || root->weights[0].value < *predictions) 
                                    ? root->weights[0].value : *predictions;
            *has_predictions = 1;
        }

        void ProcessTreeNodePrediction(NTYPE* predictions, TreeNodeElement<NTYPE> * root,
                                       unsigned char* has_predictions) const {
            for(auto it = root->weights.begin(); it != root->weights.end(); ++it) {
                predictions[it->i] = (!has_predictions[it->i] || it->value < predictions[it->i]) 
                                        ? it->value : predictions[it->i];
                has_predictions[it->i] = 1;
            }
        }

        void MergePrediction(int64_t n, NTYPE* predictions, unsigned char* has_predictions,
                             const NTYPE* predictions2, const unsigned char* has_predictions2) const {
            for(int64_t i = 0; i < n; ++i) {
                if (has_predictions2[i]) {
                    predictions[i] = has_predictions[i] && (predictions[i] < predictions2[i])
                                        ? predictions[i]
                                        : predictions2[i];
                    has_predictions[i] = 1;
                }
            }
        }

        inline void MergeOnePrediction(NTYPE* predictions, unsigned char* has_predictions,
                                       const NTYPE* predictions2, const unsigned char* has_predictions2) const {
            if (*has_predictions2) {
                *predictions = *has_predictions && (*predictions < *predictions2)
                                    ? *predictions
                                    : *predictions2;
                *has_predictions = 1;
            }
        }
};


template<typename NTYPE>
class _AggregatorMax : public _Aggregator<NTYPE>
{
    public:

        inline _AggregatorMax<NTYPE>(size_t n_trees,
                                     const int64_t& n_targets_or_classes,
                                     POST_EVAL_TRANSFORM post_transform,
                                     const std::vector<NTYPE> * base_values) :
            _Aggregator<NTYPE>(n_trees, n_targets_or_classes,
                               post_transform, base_values) { }

        inline void ProcessTreeNodePrediction1(NTYPE* predictions, TreeNodeElement<NTYPE> * root,
                                               unsigned char* has_predictions) const {
            *predictions = (!(*has_predictions) || root->weights[0].value > *predictions) 
                                    ? root->weights[0].value : *predictions;
            *has_predictions = 1;
        }

        void ProcessTreeNodePrediction(NTYPE* predictions, TreeNodeElement<NTYPE> * root,
                                       unsigned char* has_predictions) const {
            for(auto it = root->weights.begin(); it != root->weights.end(); ++it) {
                predictions[it->i] = (!has_predictions[it->i] || it->value > predictions[it->i]) 
                                        ? it->value : predictions[it->i];
                has_predictions[it->i] = 1;
            }
        }

        void MergePrediction(int64_t n, NTYPE* predictions, unsigned char* has_predictions,
                             NTYPE* predictions2, unsigned char* has_predictions2) const {
            for(int64_t i = 0; i < n; ++i) {
                if (has_predictions2[i]) {
                    predictions[i] = has_predictions[i] && (predictions[i] > predictions2[i])
                                        ? predictions[i]
                                        : predictions2[i];
                    has_predictions[i] = 1;
                }
            }
        }

        inline void MergeOnePrediction(NTYPE* predictions, unsigned char* has_predictions,
                                       const NTYPE* predictions2, const unsigned char* has_predictions2) const {
            if (*has_predictions2) {
                *predictions = *has_predictions && (*predictions > *predictions2)
                                    ? *predictions
                                    : *predictions2;
                *has_predictions = 1;
            }
        }
};


template<typename NTYPE>
class RuntimeTreeEnsembleCommonP
{
    public:

        // tree_ensemble_regressor.h
        std::vector<NTYPE> base_values_;
        int64_t n_targets_or_classes_;
        POST_EVAL_TRANSFORM post_transform_;
        AGGREGATE_FUNCTION aggregate_function_;
        int64_t n_nodes_;
        TreeNodeElement<NTYPE>* nodes_;
        std::vector<TreeNodeElement<NTYPE>*> roots_;

        int64_t max_tree_depth_;
        int64_t n_trees_;
        bool same_mode_;
        bool has_missing_tracks_;
        int omp_tree_;
        int omp_N_;

    public:

        RuntimeTreeEnsembleCommonP(int omp_tree, int omp_N);
        ~RuntimeTreeEnsembleCommonP();

        void init(
            const std::string &aggregate_function,
            py::array_t<NTYPE> base_values,
            int64_t n_targets_or_classes,
            py::array_t<int64_t> nodes_falsenodeids,
            py::array_t<int64_t> nodes_featureids,
            py::array_t<NTYPE> nodes_hitrates,
            py::array_t<int64_t> nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            py::array_t<int64_t> nodes_nodeids,
            py::array_t<int64_t> nodes_treeids,
            py::array_t<int64_t> nodes_truenodeids,
            py::array_t<NTYPE> nodes_values,
            const std::string& post_transform,
            py::array_t<int64_t> target_class_ids,
            py::array_t<int64_t> target_class_nodeids,
            py::array_t<int64_t> target_class_treeids,
            py::array_t<NTYPE> target_class_weights);

        void init_c(
            const std::string &aggregate_function,
            const std::vector<NTYPE>& base_values,
            int64_t n_targets_or_classes,
            const std::vector<int64_t>& nodes_falsenodeids,
            const std::vector<int64_t>& nodes_featureids,
            const std::vector<NTYPE>& nodes_hitrates,
            const std::vector<int64_t>& nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            const std::vector<int64_t>& nodes_nodeids,
            const std::vector<int64_t>& nodes_treeids,
            const std::vector<int64_t>& nodes_truenodeids,
            const std::vector<NTYPE>& nodes_values,
            const std::string& post_transform,
            const std::vector<int64_t>& target_class_ids,
            const std::vector<int64_t>& target_class_nodeids,
            const std::vector<int64_t>& target_class_treeids,
            const std::vector<NTYPE>& target_class_weights);

        TreeNodeElement<NTYPE> * ProcessTreeNodeLeave(
            TreeNodeElement<NTYPE> * root, const NTYPE* x_data) const;

        std::string runtime_options();
        std::vector<std::string> get_nodes_modes() const;

        int omp_get_max_threads();

        template<typename AGG>
        py::array_t<NTYPE> compute_tree_outputs_agg(py::array_t<NTYPE> X, const AGG &agg) const;
        
        py::array_t<int> debug_threshold(py::array_t<NTYPE> values) const;

        template<typename AGG>
        py::array_t<NTYPE> compute_agg(py::array_t<NTYPE> X, const AGG &agg) const;

        template<typename AGG>
        py::tuple compute_cl_agg(py::array_t<NTYPE> X, const AGG &agg) const;

    private :

        template<typename AGG>
        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                              py::array_t<int64_t>* Y, const AGG &agg) const;        
};


template<typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::RuntimeTreeEnsembleCommonP(int omp_tree, int omp_N) {
    omp_tree_ = omp_tree;
    omp_N_ = omp_N;
    nodes_ = NULL;
}


template<typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::~RuntimeTreeEnsembleCommonP() {
    if (nodes_ != NULL)
        delete [] nodes_;
}


template<typename NTYPE>
std::string RuntimeTreeEnsembleCommonP<NTYPE>::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


template<typename NTYPE>
int RuntimeTreeEnsembleCommonP<NTYPE>::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
}


template<typename NTYPE>
void RuntimeTreeEnsembleCommonP<NTYPE>::init(
            const std::string &aggregate_function,
            py::array_t<NTYPE> base_values,
            int64_t n_targets_or_classes,
            py::array_t<int64_t> nodes_falsenodeids,
            py::array_t<int64_t> nodes_featureids,
            py::array_t<NTYPE> nodes_hitrates,
            py::array_t<int64_t> nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            py::array_t<int64_t> nodes_nodeids,
            py::array_t<int64_t> nodes_treeids,
            py::array_t<int64_t> nodes_truenodeids,
            py::array_t<NTYPE> nodes_values,
            const std::string& post_transform,
            py::array_t<int64_t> target_class_ids,
            py::array_t<int64_t> target_class_nodeids,
            py::array_t<int64_t> target_class_treeids,
            py::array_t<NTYPE> target_class_weights) {

    std::vector<NTYPE> cbasevalues;
    array2vector(cbasevalues, base_values, NTYPE);

    std::vector<int64_t> tnodes_treeids;
    std::vector<int64_t> tnodes_nodeids;
    std::vector<int64_t> tnodes_featureids;
    std::vector<NTYPE> tnodes_values;
    std::vector<NTYPE> tnodes_hitrates;
    std::vector<int64_t> tnodes_truenodeids;
    std::vector<int64_t> tnodes_falsenodeids;
    std::vector<int64_t> tmissing_tracks_true;

    array2vector(tnodes_falsenodeids, nodes_falsenodeids, int64_t);
    array2vector(tnodes_featureids, nodes_featureids, int64_t);
    array2vector(tnodes_hitrates, nodes_hitrates, NTYPE);
    array2vector(tmissing_tracks_true, nodes_missing_value_tracks_true, int64_t);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);
    //nodes_modes_names_ = nodes_modes;
    array2vector(tnodes_nodeids, nodes_nodeids, int64_t);
    array2vector(tnodes_treeids, nodes_treeids, int64_t);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);
    array2vector(tnodes_values, nodes_values, NTYPE);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);

    std::vector<int64_t> ttarget_class_nodeids;
    std::vector<int64_t> ttarget_class_treeids;
    std::vector<int64_t> ttarget_class_ids;
    std::vector<NTYPE> ttarget_class_weights;
    
    array2vector(ttarget_class_ids, target_class_ids, int64_t);
    array2vector(ttarget_class_nodeids, target_class_nodeids, int64_t);
    array2vector(ttarget_class_treeids, target_class_treeids, int64_t);
    array2vector(ttarget_class_weights, target_class_weights, NTYPE);
    
    init_c(aggregate_function, cbasevalues, n_targets_or_classes,
           tnodes_falsenodeids, tnodes_featureids, tnodes_hitrates,
           tmissing_tracks_true, nodes_modes,
           tnodes_nodeids, tnodes_treeids, tnodes_truenodeids,
           tnodes_values, post_transform, ttarget_class_ids,
           ttarget_class_nodeids, ttarget_class_treeids,
           ttarget_class_weights);
}    
    
template<typename NTYPE>
void RuntimeTreeEnsembleCommonP<NTYPE>::init_c(
            const std::string &aggregate_function,
            const std::vector<NTYPE>& base_values,
            int64_t n_targets_or_classes,
            const std::vector<int64_t>& nodes_falsenodeids,
            const std::vector<int64_t>& nodes_featureids,
            const std::vector<NTYPE>& nodes_hitrates,
            const std::vector<int64_t>& nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            const std::vector<int64_t>& nodes_nodeids,
            const std::vector<int64_t>& nodes_treeids,
            const std::vector<int64_t>& nodes_truenodeids,
            const std::vector<NTYPE>& nodes_values,
            const std::string& post_transform,
            const std::vector<int64_t>& target_class_ids,
            const std::vector<int64_t>& target_class_nodeids,
            const std::vector<int64_t>& target_class_treeids,
            const std::vector<NTYPE>& target_class_weights) {

    aggregate_function_ = to_AGGREGATE_FUNCTION(aggregate_function);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    base_values_ = base_values;
    n_targets_or_classes_ = n_targets_or_classes;
    max_tree_depth_ = 1000;
    
    // additional members
    std::vector<NODE_MODE> cmodes(nodes_modes.size());
    same_mode_ = true;
    int fpos = -1;
    for(size_t i = 0; i < nodes_modes.size(); ++i) {
        cmodes[i] = to_NODE_MODE(nodes_modes[i]);
        if (cmodes[i] == NODE_MODE::LEAF)
            continue;
        if (fpos == -1) {
            fpos = (int)i;
            continue;
        }
        if (cmodes[i] != cmodes[fpos])
            same_mode_ = false;
    }
    
    // filling nodes

    n_nodes_ = nodes_treeids.size();
    nodes_ = new TreeNodeElement<NTYPE>[(int)n_nodes_];
    roots_.clear();
    std::map<TreeNodeElementId, TreeNodeElement<NTYPE>*> idi;
    size_t i;

    for (i = 0; i < nodes_treeids.size(); ++i) {
        TreeNodeElement<NTYPE> * node = nodes_ + i;
        node->id.tree_id = (int)nodes_treeids[i];
        node->id.node_id = (int)nodes_nodeids[i];
        node->feature_id = (int)nodes_featureids[i];
        node->value = nodes_values[i];
        node->hitrates = i < nodes_hitrates.size() ? nodes_hitrates[i] : -1;
        node->mode = cmodes[i];
        node->is_not_leave = node->mode != NODE_MODE::LEAF;
        node->truenode = NULL; // nodes_truenodeids[i];
        node->falsenode = NULL; // nodes_falsenodeids[i];
        node->missing_tracks = i < (size_t)nodes_missing_value_tracks_true.size()
                                    ? (nodes_missing_value_tracks_true[i] == 1 
                                            ? MissingTrack::TRUE : MissingTrack::FALSE)
                                    : MissingTrack::NONE;
        node->is_missing_track_true = node->missing_tracks == MissingTrack::TRUE;
        if (idi.find(node->id) != idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Node %d in tree %d is already there.",
                    (int)node->id.node_id, (int)node->id.tree_id);
            throw std::runtime_error(buffer);
        }
        idi.insert(std::pair<TreeNodeElementId, TreeNodeElement<NTYPE>*>(node->id, node));
    }

    TreeNodeElementId coor;
    TreeNodeElement<NTYPE> * it;
    for(i = 0; i < (size_t)n_nodes_; ++i) {
        it = nodes_ + i;
        if (!it->is_not_leave)
            continue;
        coor.tree_id = it->id.tree_id;
        coor.node_id = (int)nodes_truenodeids[i];

        auto found = idi.find(coor);
        if (found == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (truenode).",
                    (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        if (coor.node_id >= 0 && coor.node_id < n_nodes_) {
            it->truenode = found->second;
            if ((it->truenode->id.tree_id != it->id.tree_id) ||
                (it->truenode->id.node_id == it->id.node_id)) {
                char buffer[1000];
                sprintf(
                    buffer,
                    "truenode [%d] is pointing either to itself [node id=%d], either to another tree [%d!=%d-%d].",
                    (int)i, (int)it->id.node_id, (int)it->id.tree_id,
                    (int)it->truenode->id.tree_id, (int)it->truenode->id.tree_id);
                throw std::runtime_error(buffer);
            }
        }
        else it->truenode = NULL;

        coor.node_id = (int)nodes_falsenodeids[i];
        found = idi.find(coor);
        if (found == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (falsenode).",
                    (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        if (coor.node_id >= 0 && coor.node_id < n_nodes_) {
            it->falsenode = found->second;
            if ((it->falsenode->id.tree_id != it->id.tree_id) ||
                (it->falsenode->id.node_id == it->id.node_id )) {
                throw std::runtime_error("One falsenode is pointing either to itself, either to another tree.");
                char buffer[1000];
                sprintf(buffer, "falsenode [%d] is pointing either to itself [node id=%d], either to another tree [%d!=%d-%d].",
                    (int)i, (int)it->id.node_id, (int)it->id.tree_id,
                    (int)it->falsenode->id.tree_id, (int)it->falsenode->id.tree_id);
                throw std::runtime_error(buffer);
            }
        }
        else it->falsenode = NULL;
    }
    
    int64_t previous = -1;
    for(i = 0; i < (size_t)n_nodes_; ++i) {
        if ((previous == -1) || (previous != nodes_[i].id.tree_id))
            roots_.push_back(nodes_ + i);
        previous = nodes_[i].id.tree_id;
    }

    TreeNodeElementId ind;
    SparseValue<NTYPE> w;
    for (i = 0; i < target_class_nodeids.size(); i++) {
        ind.tree_id = (int)target_class_treeids[i];
        ind.node_id = (int)target_class_nodeids[i];
        if (idi.find(ind) == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (weights).", (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        w.i = target_class_ids[i];
        w.value = target_class_weights[i];
        idi[ind]->weights.push_back(w);
    }

    n_trees_ = roots_.size();
    has_missing_tracks_ = false;
    for (auto it = nodes_missing_value_tracks_true.begin();
         it != nodes_missing_value_tracks_true.end(); ++it) {
        if (*it) {
            has_missing_tracks_ = true;
            break;
        }
    }
}


template<typename NTYPE>
std::vector<std::string> RuntimeTreeEnsembleCommonP<NTYPE>::get_nodes_modes() const {
    std::vector<std::string> res;
    for(int i = 0; i < (int)n_nodes_; ++i)
        res.push_back(to_str(nodes_[i].mode));
    return res;
}


template<typename NTYPE>
template<typename AGG>
py::array_t<NTYPE> RuntimeTreeEnsembleCommonP<NTYPE>::compute_agg(py::array_t<NTYPE> X, const AGG &agg) const {
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");

    // Does not handle 3D tensors
    bool xdims1 = x_dims.size() == 1;
    int64_t stride = xdims1 ? x_dims[0] : x_dims[1];  
    int64_t N = xdims1 ? 1 : x_dims[0];

    py::array_t<NTYPE> Z(x_dims[0] * n_targets_or_classes_);

    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z, NULL, agg);
    }
    return Z;
}


template<typename NTYPE>
template<typename AGG>
py::tuple RuntimeTreeEnsembleCommonP<NTYPE>::compute_cl_agg(
        py::array_t<NTYPE> X, const AGG &agg) const {
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");

    // Does not handle 3D tensors
    bool xdims1 = x_dims.size() == 1;
    int64_t stride = xdims1 ? x_dims[0] : x_dims[1];  
    int64_t N = xdims1 ? 1 : x_dims[0];

    // Tensor* Y = context->Output(0, TensorShape({N}));
    // auto* Z = context->Output(1, TensorShape({N, class_count_}));
    py::array_t<NTYPE> Z(x_dims[0] * n_targets_or_classes_);
    py::array_t<int64_t> Y(x_dims[0]);

    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z, &Y, agg);
    }
    return py::make_tuple(Y, Z);
}


py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(py::array_t<float>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<int64_t, 1> _mutable_unchecked1(py::array_t<int64_t>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(py::array_t<double>& Z) {
    return Z.mutable_unchecked<1>();
}


template<typename NTYPE>
template<typename AGG>
void RuntimeTreeEnsembleCommonP<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                py::array_t<int64_t>* Y, const AGG &agg) const {

    // expected primary-expression before ')' token
    auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
    const NTYPE* x_data = X.data(0);

    if (n_targets_or_classes_ == 1) {
        if (N == 1) {
            NTYPE scores;
            unsigned char has_scores;
            agg.init_score(scores, has_scores);
            if (n_trees_ <= omp_tree_) {
                for (int64_t j = 0; j < n_trees_; ++j)
                    agg.ProcessTreeNodePrediction1(
                        &scores,
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        &has_scores);
            }
            else {
                std::vector<NTYPE> scores_t(n_trees_, (NTYPE)0);
                std::vector<unsigned char> has_scores_t(n_trees_, 0);
                #ifdef USE_OPENMP
                #pragma omp parallel
                #endif
                for (int64_t j = 0; j < n_trees_; ++j) {
                    agg.ProcessTreeNodePrediction1(
                        &(scores_t[j]),
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        &(has_scores_t[j]));
                }
                auto it = scores_t.cbegin();
                auto it2 = has_scores_t.cbegin();
                for(; it != scores_t.cend(); ++it, ++it2)
                    agg.MergeOnePrediction(&scores, &has_scores, &(*it), &(*it2));
            }

            agg.FinalizeScores1((NTYPE*)Z_.data(0), scores, has_scores,
                                Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else {
            if (N <= omp_N_) {
                NTYPE scores;
                unsigned char has_scores;
                size_t j;

                for (int64_t i = 0; i < N; ++i) {
                    agg.init_score(scores, has_scores);
                    for (j = 0; j < (size_t)n_trees_; ++j)
                        agg.ProcessTreeNodePrediction1(
                            &scores,
                            ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                            &has_scores);
                    agg.FinalizeScores1((NTYPE*)Z_.data(i), scores, has_scores,
                                        Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                }
            }
            else {
                NTYPE scores;
                unsigned char has_scores;
                size_t j;

                #ifdef USE_OPENMP
                #pragma omp parallel for private(j, scores, has_scores)
                #endif
                for (int64_t i = 0; i < N; ++i) {
                    agg.init_score(scores, has_scores);
                    for (j = 0; j < (size_t)n_trees_; ++j)
                        agg.ProcessTreeNodePrediction1(
                            &scores,
                            ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                            &has_scores);
                    agg.FinalizeScores1((NTYPE*)Z_.data(i), scores, has_scores,
                                        Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                }
            }
        }
    }
    else {
        if (N == 1) {
            std::vector<NTYPE> scores(n_targets_or_classes_, (NTYPE)0);
            std::vector<unsigned char> has_scores(n_targets_or_classes_, 0);

            if (n_trees_ <= omp_tree_) {
                for (int64_t j = 0; j < n_trees_; ++j)
                    agg.ProcessTreeNodePrediction(
                        scores.data(),
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        has_scores.data());
                agg.FinalizeScores(scores, has_scores, (NTYPE*)Z_.data(0), -1,
                                   Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(0));
            }
            else {            
                #ifdef USE_OPENMP
                #pragma omp parallel
                #endif
                {
                    std::vector<NTYPE> private_scores(n_targets_or_classes_, (NTYPE)0);
                    std::vector<unsigned char> private_has_scores(n_targets_or_classes_, 0);
                    #ifdef USE_OPENMP
                    #pragma omp for
                    #endif
                    for (int64_t j = 0; j < n_trees_; ++j) {
                        agg.ProcessTreeNodePrediction(
                            private_scores.data(),
                            ProcessTreeNodeLeave(roots_[j], x_data),
                            private_has_scores.data());
                    }

                    #ifdef USE_OPENMP
                    #pragma omp critical
                    #endif
                    agg.MergePrediction(n_targets_or_classes_,
                        &(scores[0]), &(has_scores[0]),
                        private_scores.data(), private_has_scores.data());
                }
                
                agg.FinalizeScores(scores, has_scores, (NTYPE*)Z_.data(0), -1,
                                   Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(0));
            }
        }
        else {
            if (N <= omp_N_) {
                std::vector<NTYPE> scores(n_targets_or_classes_);
                std::vector<unsigned char> has_scores(n_targets_or_classes_);
                size_t j;

                for (int64_t i = 0; i < N; ++i) {
                    std::fill(scores.begin(), scores.end(), (NTYPE)0);
                    std::fill(has_scores.begin(), has_scores.end(), 0);
                    for (j = 0; j < roots_.size(); ++j)
                        agg.ProcessTreeNodePrediction(
                            scores.data(),
                            ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                            has_scores.data());
                    agg.FinalizeScores(scores, has_scores,
                                       (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                       Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                }
            }
            else {
                #ifdef USE_OPENMP
                #pragma omp parallel
                #endif
                {
                    std::vector<NTYPE> scores(n_targets_or_classes_);
                    std::vector<unsigned char> has_scores(n_targets_or_classes_);
                    size_t j;

                    #ifdef USE_OPENMP
                    #pragma omp for
                    #endif
                    for (int64_t i = 0; i < N; ++i) {
                        std::fill(scores.begin(), scores.end(), (NTYPE)0);
                        std::fill(has_scores.begin(), has_scores.end(), 0);
                        for (j = 0; j < roots_.size(); ++j)
                            agg.ProcessTreeNodePrediction(
                                scores.data(),
                                ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                                has_scores.data());
                        agg.FinalizeScores(scores, has_scores,
                                           (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                           Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                    }
                }
            }
        }
    }
}
              


#define TREE_FIND_VALUE(CMP) \
    if (has_missing_tracks_) { \
        while (root->is_not_leave) { \
            val = x_data[root->feature_id]; \
            root = (val CMP root->value || \
                    (root->is_missing_track_true && _isnan_(val) )) \
                        ? root->truenode : root->falsenode; \
        } \
    } \
    else { \
        while (root->is_not_leave) { \
            val = x_data[root->feature_id]; \
            root = val CMP root->value ? root->truenode : root->falsenode; \
        } \
    }


template<typename NTYPE>
TreeNodeElement<NTYPE> * 
        RuntimeTreeEnsembleCommonP<NTYPE>::ProcessTreeNodeLeave(
            TreeNodeElement<NTYPE> * root, const NTYPE* x_data) const {
    NTYPE val;
    if (same_mode_) {
        switch(root->mode) {
            case NODE_MODE::BRANCH_LEQ:
                if (has_missing_tracks_) {
                    while (root->is_not_leave) {
                        val = x_data[root->feature_id];
                        root = (val <= root->value ||
                                (root->is_missing_track_true && _isnan_(val) ))
                                    ? root->truenode : root->falsenode;
                    }
                }
                else {
                    while (root->is_not_leave) {
                        val = x_data[root->feature_id];
                        root = val <= root->value ? root->truenode : root->falsenode;
                    }
                }
                break;
            case NODE_MODE::BRANCH_LT:
                TREE_FIND_VALUE(<)
                break;
            case NODE_MODE::BRANCH_GTE:
                TREE_FIND_VALUE(>=)
                break;
            case NODE_MODE::BRANCH_GT:
                TREE_FIND_VALUE(>)
                break;
            case NODE_MODE::BRANCH_EQ:
                TREE_FIND_VALUE(==)
                break;
            case NODE_MODE::BRANCH_NEQ:
                TREE_FIND_VALUE(!=)
                break;
            case NODE_MODE::LEAF:
                break;
            default: {
                std::ostringstream err_msg;
                err_msg << "Invalid mode of value: " << static_cast<std::underlying_type<NODE_MODE>::type>(root->mode);
                throw std::runtime_error(err_msg.str());
            }
        }
    }
    else {  // Different rules to compare to node thresholds.
        NTYPE threshold;
        while (root->is_not_leave) {
            val = x_data[root->feature_id];
            threshold = root->value;
            switch (root->mode) {
                case NODE_MODE::BRANCH_LEQ:
                    root = val <= threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_LT:
                    root = val < threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_GTE:
                    root = val >= threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_GT:
                    root = val > threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_EQ:
                    root = val == threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_NEQ:
                    root = val != threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                default: {
                    std::ostringstream err_msg;
                    err_msg << "Invalid mode of value: "
                            << static_cast<std::underlying_type<NODE_MODE>::type>(root->mode);
                    throw std::runtime_error(err_msg.str());
                }
            }
        }      
    }
    return root;
}


template<typename NTYPE>
py::array_t<int> RuntimeTreeEnsembleCommonP<NTYPE>::debug_threshold(
        py::array_t<NTYPE> values) const {
    std::vector<int> result(values.size() * n_nodes_);
    const NTYPE* x_data = values.data(0);
    const NTYPE* end = x_data + values.size();
    const NTYPE* pv;
    auto itb = result.begin();
    auto nodes_end = nodes_ + n_nodes_;
    for(auto it = nodes_; it != nodes_end; ++it)
        for(pv=x_data; pv != end; ++pv, ++itb)
            *itb = *pv <= it->value ? 1 : 0;
    std::vector<ssize_t> shape = { n_nodes_, values.size() };
    std::vector<ssize_t> strides = { (ssize_t)(values.size()*sizeof(int)),
                                     (ssize_t)sizeof(int) };
    return py::array_t<NTYPE>(
        py::buffer_info(
            &result[0],
            sizeof(NTYPE),
            py::format_descriptor<NTYPE>::format(),
            2,
            shape,                                   /* shape of the matrix       */
            strides                                  /* strides for each axis     */
        ));
}


template<typename NTYPE>
template<typename AGG>
py::array_t<NTYPE> RuntimeTreeEnsembleCommonP<NTYPE>::compute_tree_outputs_agg(py::array_t<NTYPE> X, const AGG &agg) const {
    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");

    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];

    std::vector<NTYPE> result(N * roots_.size());
    const NTYPE* x_data = X.data(0);
    auto itb = result.begin();

    for (int64_t i=0; i < N; ++i) {  //for each class or target
        int64_t current_weight_0 = i * stride;
        for (size_t j = 0; j < roots_.size(); ++j, ++itb) {
            std::vector<NTYPE> scores(n_targets_or_classes_, (NTYPE)0);
            std::vector<unsigned char> has_scores(n_targets_or_classes_, 0);
            agg.ProcessTreeNodePrediction(
                    scores.data(),
                    ProcessTreeNodeLeave(roots_[j], x_data + current_weight_0),
                    has_scores.data());
            *itb = scores[0];
        }
    }

    std::vector<ssize_t> shape = { (ssize_t)N, (ssize_t)roots_.size() };
    std::vector<ssize_t> strides = { (ssize_t)(roots_.size()*sizeof(NTYPE)),
                                     (ssize_t)sizeof(NTYPE) };
    return py::array_t<NTYPE>(
        py::buffer_info(
            &result[0],
            sizeof(NTYPE),
            py::format_descriptor<NTYPE>::format(),
            2,
            shape,                                   /* shape of the matrix       */
            strides                                  /* strides for each axis     */
        ));
}
