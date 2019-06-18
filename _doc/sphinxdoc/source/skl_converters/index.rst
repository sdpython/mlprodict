
Visual Representation of scikit-learn models
============================================

:epkg:`sklearn-onnx` converts many models from
:epkg:`scikit-learn` into :epkg:`ONNX`. Every of
them is a graph made of :epkg:`ONNX` mathematical functions
(see :ref:`l-onnx-runtime-operators`,
:epkg:`ONNX Operators`, :epkg:`ONNX ML Operators`).
The following sections display a visual representation
of each converted model. Every graph
represents one ONNX graphs obtained after a model
is fitted. The structure may change is the model is trained
again.

.. toctree::
    :maxdepth: 1

    skl2onnx_calibration
    skl2onnx_cluster
    skl2onnx_decomposition
    skl2onnx_ensemble
    skl2onnx_feature_selection
    skl2onnx_impute
    skl2onnx_linear_model
    skl2onnx_mixture
    skl2onnx_multiclass
    skl2onnx_naive_bayes
    skl2onnx_neighbors
    skl2onnx_neural_network
    skl2onnx_preprocessing
    skl2onnx_svm
    skl2onnx_tree
