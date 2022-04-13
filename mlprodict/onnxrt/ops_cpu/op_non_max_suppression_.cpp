// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef SKIP_PYTHON
//#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include <numpy/arrayobject.h>
#include "op_common_.hpp"

#if USE_OPENMP
#include <omp.h>
#endif

#include <memory>
#include <queue>

namespace py = pybind11;
#endif

//////////
// classes
//////////

#define HelperMin(a, b) (a < b ? a : b)
#define HelperMax(a, b) (a > b ? a : b)


struct PrepareContext {
    const float* boxes_data_ = nullptr;
    int64_t boxes_size_ = 0ll;
    const float* scores_data_ = nullptr;
    int64_t scores_size_ = 0ll;
    // The below are ptrs since they cab be device specific
    const int64_t* max_output_boxes_per_class_ = nullptr;
    const float* score_threshold_ = nullptr;
    const float* iou_threshold_ = nullptr;
    int64_t num_batches_ = 0;
    int64_t num_classes_ = 0;
    int num_boxes_ = 0;
};


struct SelectedIndex {
    SelectedIndex(int64_t batch_index, int64_t class_index, int64_t box_index)
      : batch_index_(batch_index), class_index_(class_index), box_index_(box_index) {}
    SelectedIndex() = default;
    int64_t batch_index_ = 0;
    int64_t class_index_ = 0;
    int64_t box_index_ = 0;
};


inline void MaxMin(float lhs, float rhs, float& min, float& max) {
    if (lhs >= rhs) {
        min = rhs;
        max = lhs;
    } else {
        min = lhs;
        max = rhs;
    }
}


inline bool SuppressByIOU(const float* boxes_data, int64_t box_index1, int64_t box_index2,
                          int64_t center_point_box, float iou_threshold) {
    float x1_min{};
    float y1_min{};
    float x1_max{};
    float y1_max{};
    float x2_min{};
    float y2_min{};
    float x2_max{};
    float y2_max{};
    float intersection_x_min{};
    float intersection_x_max{};
    float intersection_y_min{};
    float intersection_y_max{};

    const float* box1 = boxes_data + 4 * box_index1;
    const float* box2 = boxes_data + 4 * box_index2;
    // center_point_box_ only support 0 or 1
    if (0 == center_point_box) {
    // boxes data format [y1, x1, y2, x2],
        MaxMin(box1[1], box1[3], x1_min, x1_max);
        MaxMin(box2[1], box2[3], x2_min, x2_max);

        intersection_x_min = HelperMax(x1_min, x2_min);
        intersection_x_max = HelperMin(x1_max, x2_max);
        if (intersection_x_max <= intersection_x_min)
            return false;

        MaxMin(box1[0], box1[2], y1_min, y1_max);
        MaxMin(box2[0], box2[2], y2_min, y2_max);
        intersection_y_min = HelperMax(y1_min, y2_min);
        intersection_y_max = HelperMin(y1_max, y2_max);
        if (intersection_y_max <= intersection_y_min)
            return false;
    }
    else {
        // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
        float box1_width_half = box1[2] / 2;
        float box1_height_half = box1[3] / 2;
        float box2_width_half = box2[2] / 2;
        float box2_height_half = box2[3] / 2;

        x1_min = box1[0] - box1_width_half;
        x1_max = box1[0] + box1_width_half;
        x2_min = box2[0] - box2_width_half;
        x2_max = box2[0] + box2_width_half;

        intersection_x_min = HelperMax(x1_min, x2_min);
        intersection_x_max = HelperMin(x1_max, x2_max);
        if (intersection_x_max <= intersection_x_min)
            return false;

        y1_min = box1[1] - box1_height_half;
        y1_max = box1[1] + box1_height_half;
        y2_min = box2[1] - box2_height_half;
        y2_max = box2[1] + box2_height_half;

        intersection_y_min = HelperMax(y1_min, y2_min);
        intersection_y_max = HelperMin(y1_max, y2_max);
        if (intersection_y_max <= intersection_y_min)
            return false;
    }

    const float intersection_area = 
        (intersection_x_max - intersection_x_min) *
        (intersection_y_max - intersection_y_min);

    if (intersection_area <= .0f)
        return false;

    const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
    const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
    const float union_area = area1 + area2 - intersection_area;

    if (area1 <= .0f || area2 <= .0f || union_area <= .0f)
        return false;

    const float intersection_over_union = intersection_area / union_area;
    return intersection_over_union > iou_threshold;
}


struct BoxInfoPtr {
    float score_{};
    int64_t index_{};

    BoxInfoPtr() = default;
    explicit BoxInfoPtr(float score, int64_t idx) : score_(score), index_(idx) {}
    inline bool operator<(const BoxInfoPtr& rhs) const {
        return score_ < rhs.score_ || (score_ == rhs.score_ && index_ > rhs.index_);
    }
};


class RuntimeNonMaxSuppression {
    private:

        int64_t center_point_box_;

    public:

        void init(const int64_t& center_point_box) {
            center_point_box_ = center_point_box;
        }

        py::array_t<int64_t> compute(const py::array_t<float, py::array::c_style | py::array::forcecast>& boxes_tensor,
                                     const py::array_t<float, py::array::c_style | py::array::forcecast>& scores_tensor,
                                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& max_output_boxes_per_class_tensor,
                                     const py::array_t<float, py::array::c_style | py::array::forcecast>& iou_threshold_tensor,
                                     const py::array_t<float, py::array::c_style | py::array::forcecast>& score_threshold_tensor) const {
            py::array_t<int64_t> result;
            Compute(result, boxes_tensor, scores_tensor,
                    max_output_boxes_per_class_tensor,
                    iou_threshold_tensor, score_threshold_tensor);
            return result;
        }

    protected:

        void Compute(py::array_t<int64_t>& result,
                     const py::array_t<float>& boxes_tensor,
                     const py::array_t<float>& scores_tensor,
                     const py::array_t<int64_t>& max_output_boxes_per_class_tensor,
                     const py::array_t<float>& iou_threshold_tensor,
                     const py::array_t<float>& score_threshold_tensor) const {
            PrepareContext pc;
            PrepareCompute(pc, boxes_tensor, scores_tensor,
                           max_output_boxes_per_class_tensor,
                           iou_threshold_tensor, score_threshold_tensor);

            int64_t max_output_boxes_per_class = 0;
            float iou_threshold = .0f;
            float score_threshold = .0f;
            
            GetThresholdsFromInputs(pc, max_output_boxes_per_class, iou_threshold, score_threshold);

            if (max_output_boxes_per_class_tensor.ndim() == 0) {
                result = py::array_t<int64_t>();
                return;
            }

            const auto* const boxes_data = pc.boxes_data_;
            const auto* const scores_data = pc.scores_data_;
            const auto center_point_box = center_point_box_;

            std::vector<SelectedIndex> selected_indices;
            std::vector<BoxInfoPtr> selected_boxes_inside_class;
            selected_boxes_inside_class.reserve(
                std::min<size_t>(static_cast<size_t>(max_output_boxes_per_class), pc.num_boxes_));

            for (int64_t batch_index = 0; batch_index < pc.num_batches_; ++batch_index) {
                for (int64_t class_index = 0; class_index < pc.num_classes_; ++class_index) {
                    int64_t box_score_offset = (batch_index * pc.num_classes_ + class_index) * pc.num_boxes_;
                    const float* batch_boxes = boxes_data + (batch_index * pc.num_boxes_ * 4);
                    std::vector<BoxInfoPtr> candidate_boxes;
                    candidate_boxes.reserve(pc.num_boxes_);

                    // Filter by score_threshold_
                    const auto* class_scores = scores_data + box_score_offset;
                    if (pc.score_threshold_ != nullptr) {
                        for (int64_t box_index = 0; box_index < pc.num_boxes_; ++box_index, ++class_scores) {
                            if (*class_scores > score_threshold) {
                                candidate_boxes.emplace_back(*class_scores, box_index);
                            }
                        }
                    } 
                    else {
                        for (int64_t box_index = 0; box_index < pc.num_boxes_; ++box_index, ++class_scores) {
                            candidate_boxes.emplace_back(*class_scores, box_index);
                        }
                    }
                    std::priority_queue<BoxInfoPtr, std::vector<BoxInfoPtr>> sorted_boxes(
                        std::less<BoxInfoPtr>(), std::move(candidate_boxes));

                    selected_boxes_inside_class.clear();
                    // Get the next box with top score, filter by iou_threshold
                    while (!sorted_boxes.empty() && static_cast<int64_t>(selected_boxes_inside_class.size()) < max_output_boxes_per_class) {
                        const BoxInfoPtr& next_top_score = sorted_boxes.top();

                        bool selected = true;
                        // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
                        for (const auto& selected_index : selected_boxes_inside_class) {
                            if (SuppressByIOU(batch_boxes, next_top_score.index_, selected_index.index_, center_point_box, iou_threshold)) {
                                selected = false;
                                break;
                            }
                        }

                        if (selected) {
                            selected_boxes_inside_class.push_back(next_top_score);
                            selected_indices.emplace_back(batch_index, class_index, next_top_score.index_);
                        }
                        sorted_boxes.pop();
                    }
                }
            }

            const auto num_selected = selected_indices.size();
            result = py::array_t<int64_t>(num_selected * sizeof(SelectedIndex) / sizeof(int64_t));
            memcpy((int64_t*)result.data(), selected_indices.data(),
                   num_selected * sizeof(SelectedIndex));
        }

        void GetThresholdsFromInputs(
                        const PrepareContext& pc, int64_t& max_output_boxes_per_class,
                        float& iou_threshold, float& score_threshold) const {
            if (pc.max_output_boxes_per_class_ != nullptr)
                max_output_boxes_per_class = std::max<int64_t>(*pc.max_output_boxes_per_class_, 0);

            if (pc.iou_threshold_ != nullptr)
                iou_threshold = *pc.iou_threshold_;

            if (pc.score_threshold_ != nullptr)
                score_threshold = *pc.score_threshold_;
        }    

        void PrepareCompute(
                PrepareContext& pc,
                const py::array_t<float>& boxes_tensor,
                const py::array_t<float>& scores_tensor,
                const py::array_t<int64_t>& max_output_boxes_per_class_tensor,
                const py::array_t<float>& iou_threshold_tensor,
                const py::array_t<float>& score_threshold_tensor) const {
            pc.boxes_data_ = boxes_tensor.data();
            pc.scores_data_ = scores_tensor.data();

            if (max_output_boxes_per_class_tensor.ndim() != 0)
                pc.max_output_boxes_per_class_ = max_output_boxes_per_class_tensor.data();
            if (iou_threshold_tensor.ndim() != 0)
                pc.iou_threshold_ = iou_threshold_tensor.data();
            if (score_threshold_tensor.ndim() != 0)
                pc.score_threshold_ = score_threshold_tensor.data();

            pc.boxes_size_ = boxes_tensor.size();
            pc.scores_size_ = scores_tensor.size();

            const auto& boxes_dims = boxes_tensor.shape();
            const auto& scores_dims = scores_tensor.shape();

            pc.num_batches_ = boxes_dims[0];
            pc.num_classes_ = scores_dims[1];
            pc.num_boxes_ = (int) boxes_dims[1];
        }
};


/////////
// python
/////////


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_non_max_suppression_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator NonMaxSuppression."
    #else
    R"pbdoc(Implements runtime for operator NonMaxSuppression. The code is inspired from
`non_max_suppression.cc
<https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeNonMaxSuppression> cli (m, "RuntimeNonMaxSuppression",
        R"pbdoc(Implements runtime for operator NonMaxSuppression. The code is inspired from
`non_max_suppression.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cli.def(py::init<>());
    cli.def("init", &RuntimeNonMaxSuppression::init, "initialization", py::arg("center_point_box"));

    cli.def("compute", &RuntimeNonMaxSuppression::compute, "Computes NonMaxSuppression.",
            py::arg("boxes"), py::arg("scores"),
            py::arg("max_output_boxes_per_class"),
            py::arg("iou_threshold"), py::arg("score_threshold"));
}

#endif
