
.. _l-HISTORY:

=======
History
=======

current - 2019-12-08 - 0.00Mb
=============================

* `85`: Implements a precompiled run method in OnnxInference (runtime='python_compiled') (2019-12-07)
* `84`: Automatically creates files to profile time_predict function in the benchmark with py-spy (2019-12-04)
* `82`: Function translate_fct2onnx: use of opset_version (2019-12-04)
* `83`: ONNX: includes experimental operators in the benchmark (2019-12-04)
* `81`: ONNX benchmark: track_score returns scores equal to 0 or 1 (unexpected) (2019-12-04)
* `77`: Improves ONNX benchmark to measure zipmap impact. (2019-12-03)
* `80`: ONNX: extend benchmark to decision_function for some models (2019-12-03)
* `75`: ONNX: fix random_state whevever it is available when running benchmark (2019-11-27)
* `76`: Implements ArgMax 12, ArgMax 12 (python onnx runtime) (2019-11-27)

0.3.765 - 2019-11-21 - 0.22Mb
=============================

* `59`: ONNX: Investigate kmeans and opset availability. (2019-11-21)
* `66`: ONNX: improves speed of python runtime for decision trees (2019-11-19)
* `74`: Function _modify_dimension should return the same dataset if called the same parameter (even if it uses random functions) (2019-11-15)
* `73`: ONNX: fix links on benchmark page (opset is missing) (2019-11-07)
* `72`: ONNX: support of sparse tensor for a unary and binary python operators (2019-11-06)
* `71`: ONNX: add operator Constant (2019-11-06)
* `67`: ONNX: improves speed of svm regressor (2019-11-06)
* `70`: ONNX: write tools to test convervsion for models in scikit-learn examples (2019-10-29)
* `65`: ONNX: investigate discrepencies for k-NN (2019-10-28)
* `69`: ONNX: side by side should work by name and not by positions (2019-10-23)
* `68`: ONNX: improves speed of SGDClassifier (2019-10-23)
* `61`: Implements a function to create a benchmark based on asv (ONNX) (2019-10-17)
* `63`: Export asv results to csv (ONNX) + command line (2019-10-11)
* `64`: Add an example with lightgbm and categorical variables (ONNX) (2019-10-07)
* `62`: Implements command line for the asv benchmark (ONNX) (2019-10-04)
* `60`: Improve lightgbm converter (ONNX) (2019-09-30)
* `58`: Fix table checking model, merge is wrong in documentation (2019-09-20)

0.2.542 - 2019-09-15 - 0.59Mb
=============================

* `57`: ONNX: handles dataframe when converting a model (2019-09-15)
* `56`: ONNX: implements cdist operator (2019-09-12)
* `54`: ONNX: fix summary, it produces multiple row when model are different when opset is different (2019-09-12)
* `51`: ONNX: measure the time performance obtained by using optimization (2019-09-11)
* `52`: ONNC-cli: add a command line to optimize an onnx model (2019-09-10)
* `49`: ONNX optimization: remove redundant subparts of a graph (2019-09-09)
* `48`: ONNX optimization: reduce the number of Identity nodes (2019-09-09)
* `47`: Implements statistics on onnx graph and sklearn models, add them to the documentation (2019-09-06)
* `46`: Implements KNearestNeibhorsRegressor supporting batch mode (ONNX) (2019-08-31)
* `45`: KNearestNeighborsRegressor (2019-08-30)
* `44`: Add an example to look into the performance of every node for a particular dataset (2019-08-30)
* `43`: LGBMClassifier has wrong shape (2019-08-29)

0.2.452 - 2019-08-28 - 0.13Mb
=============================

* `42`: Adds a graph which visually summarize the validating benchmark (ONNX). (2019-08-27)
* `41`: Enables to test multiple number of features at the same time (ONNX) (2019-08-27)
* `40`: Add a parameter to change the number of featuress when validating a model (ONNX). (2019-08-26)
* `39`: Add a parameter to dump all models even if they don't produce errors when being validated (ONNX) (2019-08-26)
* `24`: support double for TreeEnsembleClassifier (python runtime ONNX) (2019-08-23)
* `38`: See issue on onnxmltools. https://github.com/onnx/onnxmltools/issues/321 (2019-08-19)
* `35`: Supports parameter time_kwargs in the command line (ONNX) (2019-08-09)
* `34`: Add intervals when measuring time ratios between scikit-learn and onnx (ONNX) (2019-08-09)
* `31`: Implements shape inference for the python runtime (ONNX) (2019-08-06)
* `15`: Tells operator if the execution can be done inplace for unary operators (ONNX). (2019-08-06)
* `27`: Bug fix (2019-08-02)
* `23`: support double for TreeEnsembleRegressor (python runtime ONNX) (2019-08-02)

0.2.363 - 2019-08-01 - 0.11Mb
=============================

* `26`: Tests all converters in separate processeses to make it easier to catch crashes (2019-08-01)
* `25`: Ensures operator clip returns an array of the same type (ONNX Python Runtime) (2019-07-30)
* `22`: Implements a function to shake an ONNX model and test float32 conversion (2019-07-28)
* `21`: Add customized converters (2019-07-28)
* `20`: Enables support for TreeEnsemble operators in python runtime (ONNX). (2019-07-28)
* `19`: Enables support for SVM operators in python runtime (ONNX). (2019-07-28)
* `16`: fix documentation, visual graph are not being rendered in notebooks (2019-07-23)
* `18`: implements python runtime for SVM (2019-07-20)

0.2.272 - 2019-07-15 - 0.09Mb
=============================

* `17`: add a mechanism to use ONNX with double computation (2019-07-15)
* `13`: add automated benchmark of every scikit-learn operator in the documentation (2019-07-05)
* `12`: implements a way to measure time for each node of the ONNX graph (2019-07-05)
* `11`: implements a better ZipMap node based on dedicated container (2019-07-05)
* `8`: implements runtime for decision tree (2019-07-05)
* `7`: implement python runtime for scaler, pca, knn, kmeans (2019-07-05)
* `10`: implements full runtime with onnxruntime not node by node (2019-06-16)
* `9`: implements a onnxruntime runtime (2019-06-16)
* `6`: first draft of a python runtime for onnx (2019-06-15)
* `5`: change style highlight-ipython3 (2018-01-05)
