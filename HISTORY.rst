
.. _l-HISTORY:

=======
History
=======

current - 2021-01-02 - 0.00Mb
=============================

* `200`: Add support for bfloat16 (2020-12-30)
* `201`: Fixes #200, add support for float16 (2020-12-30)
* `199`: Fix unit tests recently failing due to onnxruntime update. (2020-12-15)

0.4.1352 - 2020-12-11 - 1.42Mb
==============================

* `196`: Fixes operator Slice for opset 9 (2020-12-11)
* `198`: Fixes #197, add function to plot onnx graph with matplotlib (2020-12-09)
* `197`: Add a function to plot an onnx graph into matplotlib (2020-12-09)
* `195`: Fixes #194, add function to add an operator in the graph (2020-12-08)
* `194`: Add a function to insert a cast operator between two nodes (2020-12-08)
* `193`: Improves notebook coverage, update CI (2020-11-29)
* `192`: Fixes #191, improves performance of TreeEnsemble (2020-11-28)
* `191`: Improves performance of TreeEnsemble (2020-11-28)
* `190`: Fixes #189, parallelization of Einsum (2020-11-17)
* `189`: Introduce parallelization in experimental einsum implementation (2020-11-17)
* `188`: Fixes #187, custom implementation for operator Einsum (2020-11-15)
* `187`: Custom implementation for operator Einsum (2020-11-15)
* `186`: Fixes #185, add operator LessOrEqual (2020-11-15)
* `185`: Add operator LessOrEqual (2020-11-15)
* `181`: Fix converter xgboost when ntree_limit is set up (2020-11-14)
* `184`: Fixes #183, fix missing parameter black_op in OnnxPipeline (2020-11-07)
* `183`: Fix error in OnnxPipeline, parameter black_op not found (2020-11-07)
* `182`: Fixes #178, fix xgboost issue with ntree_limit (2020-11-07)
* `178`: Fixes unit test testing OnnxConv (issue with shapes) (2020-11-07)
* `180`: Fixes #179, fix guess_schema_from_data for categories (2020-11-03)
* `179`: guess_schema_data_type fails with category in dataframe (2020-11-03)
* `176`: Fixes #175, add operator dropout (2020-09-29)
* `175`: Add operator Dropout (2020-09-29)
* `174`: Add support for ReduceSum >= 13 (2020-09-21)
* `173`: Fixes #172, add runtime for operator MaxPool (2020-09-16)
* `172`: Add runtime for operator MaxPool (2020-09-16)
* `171`: Fixes #170, add operator Pad (2020-09-10)
* `170`: Add runtime for operator Pad (2020-09-10)

0.4.1259 - 2020-09-03 - 1.32Mb
==============================

* `169`: fix compiling issue with ubuntu 16.04 (2020-09-03)
* `167`: Add runtime for Operator Or (2020-08-25)
* `166`: Add runtime for operator And (2020-08-25)
* `165`: Add runtime for operator GreaterOrEqual (2020-08-25)
* `164`: Add runtime for operator If (2020-08-25)
* `163`: Add runtime for operator Unsqueeze (2020-08-25)
* `162`: Add runtime for operator Split (2020-08-25)
* `161`: Add support for disable_optimisation (2020-08-12)
* `160`: Fixes #159, add operator ConvTranspose, refactoring. (2020-08-07)
* `159`: Implements runtime for ConvTranspose (2020-08-07)
* `158`: Fixes benchmark import issues (2020-08-03)
* `157`: Simplify scenarios, reduce time for benchmark. (2020-08-02)
* `156`: Fixes #155, improves documentation (2020-08-02)
* `155`: Fixes API on documentation (2020-08-02)
* `154`: Fixes y_train dtype for most of the problems. Fixes subproblems with GridSearchCV (2020-07-31)
* `153`: Fixes #152, set set n_jobs to the number of CPU (2020-07-31)
* `152`: Set n_jobs to the number of core - 1 when doing benchmark (2020-07-31)
* `151`: Force operator Conv to use continuous array (2020-07-30)
* `150`: Fixes nan issue in operator conv (2020-07-29)
* `147`: Fixes #145, #150, shape inference for operator Conv (2020-07-29)
* `145`: Fixes missing shape inference for operator conv (2020-07-29)
* `149`: Fixes #148, add operator Atan (2020-07-22)
* `148`: Add operator atan (2020-07-22)
* `146`: Fixes #144, add operator GlobalAveragePool (2020-07-21)
* `144`: Implements operator GlobalAveragePool (2020-07-21)
* `143`: Fixes #142, add operator BatchNormalization (2020-07-21)
* `142`: Implement python runtime for operator BatchNormalization (2020-07-21)
* `141`: Fixes #140, add runtime for QuantizeLinear, DequantizeLinear (2020-07-20)
* `140`: Implement runtime for QuantizeLinear, DequantizeLinear (2020-07-20)

0.4.1204 - 2020-07-09 - 0.31Mb
==============================

* `139`: Add runtime for operator EyeLike (2020-07-08)
* `138`: Add code to register custom python operator (2020-07-08)
* `137`: Remove parameter dtype (onnx conversion) (2020-07-08)
* `136`: Add parameter reshape to OnnxTransformer (2020-07-03)
* `135`: Add a function to change the first dimension output (ONNX). (2020-07-03)
* `133`: Implements runtime for operator Gather (ONNX) (2020-06-18)
* `132`: Add operator StringNormalizer, Tokenizer, TfidfVectorizer (ONNX) (2020-06-15)
* `131`: Add custom operator solve (2020-06-12)
* `130`: Add operator Erf (ONNX) (2020-06-11)
* `129`: Add operator Einsum (ONNX) (2020-06-11)
* `128`: Fixes #127, implements OnnxPipeline, train, convert at each step (2020-06-08)
* `127`: Implements a pipeline which replaces early stages by onnx (2020-06-08)

0.3.1129 - 2020-06-04 - 0.29Mb
==============================

* `123`: Enables opset 12 (ONNX) (2020-06-04)
* `117`: Support for op_version in onnx grammar (2020-06-04)

0.3.1108 - 2020-05-20 - 0.29Mb
==============================

* `126`: Fix xgboost converter for xgboost >= 1.0 (2020-05-18)
* `125`: Refactor rewritten sklearn operators (2020-05-18)
* `124`: Fixes #122, capture standard C ouptput with dump_data_model, first step for #123 (2020-05-16)
* `122`: Captures C output when calling dump_data_and_model (2020-05-16)

0.3.1082 - 2020-05-01 - 2.84Mb
==============================

* `121`: Add function to convert array to bytes and bytes to array (onnx tensor) (2020-04-30)
* `120`: Fix discrepencies for SVM classifier (ONNX) (2020-04-30)
* `119`: Keep order in topk implementation (2020-04-17)
* `118`: opset is not propagated in OnnxTransformer (2020-04-09)

0.3.1070 - 2020-04-07 - 0.29Mb
==============================

* `115`: Add a function to replay a benchmark when this one was dumped (more accurate) (2020-04-06)
* `116`: Makes ZipMapDictionary picklable (2020-03-30)
* `114`: Add more parameters to specify benchmark time (2020-03-30)
* `113`: Add operators for opset 12 (2020-03-26)
* `112`: Number of feature is wrong for problem num-tr-clus (2020-03-20)

0.3.1029 - 2020-03-17 - 0.28Mb
==============================

* `111`: Reduce the number of allocation in TreeEnsemble when it is parallelized (cache) (2020-03-13)
* `110`: Implements runtime for operator Constant-12 (2020-03-06)
* `109`: Generate a benchmark with asv to compare different runtime. Update modules in asv. (2020-03-06)
* `108`: Add a function to reduce the memory footprint (2020-02-25)
* `106`: Add operator Neg (2020-02-25)
* `101`: Fix DecisionTreeClassifier disappearance on the benchmark graph (2020-02-25)
* `107`: Add operator IsNaN (2020-02-24)
* `105`: Support string labels for Linear, TreeEnsemble, SVM classifiers. (2020-02-24)
* `104`: Enable / disable parallelisation in topk (2020-02-23)
* `103`: Implements plot benchmark ratio depending on two parameters (2020-02-22)
* `102`: Fix conversion for xgboost 1.0 (2020-02-21)

0.3.975 - 2020-02-19 - 0.28Mb
=============================

* `100`: add notebook on TreeEnsemble (2020-02-19)
* `99`: Fixes #93, use same code for TreeEnsembleClassifier and TreeEnsembleRegression (2020-02-19)
* `93`: Use pointer for TreeClassifier (2020-02-19)
* `98`: mlprodict i broken after onnxruntime, skl2onnx update (2020-02-15)
* `97`: Add runtime for operator Conv (2020-01-24)
* `96`: Fixes #97, add runtime for operator Conv (2020-01-24)
* `95`: Fix OnnxInference where an output and an operator share the same name (2020-01-15)
* `94`: Raw scores are always positive for TreeEnsembleClassifier (binary) (2020-01-13)
* `90`: Implements a C++ runtime for topk (2019-12-17)
* `86`: Use pointers to replace treeindex in tree ensemble cpp runtime (2019-12-17)
* `92`: Implements a C++ version of  ArrayFeatureExtractor (2019-12-14)
* `89`: Implements a function which extracts some informations on the models (2019-12-14)
* `88`: Fix bug in runtime of GatherElements (2019-12-14)

0.3.853 - 2019-12-13 - 0.24Mb
=============================

* `87`: Add converter for HistGradientBoostRegressor (2019-12-09)
* `85`: Implements a precompiled run method in OnnxInference (runtime='python_compiled') (2019-12-07)
* `84`: Automatically creates files to profile time_predict function in the benchmark with py-spy (2019-12-04)
* `83`: ONNX: includes experimental operators in the benchmark (2019-12-04)
* `82`: Function translate_fct2onnx: use of opset_version (2019-12-04)
* `81`: ONNX benchmark: track_score returns scores equal to 0 or 1 (unexpected) (2019-12-04)
* `80`: ONNX: extend benchmark to decision_function for some models (2019-12-03)
* `77`: Improves ONNX benchmark to measure zipmap impact. (2019-12-03)
* `76`: Implements ArgMax 12, ArgMax 12 (python onnx runtime) (2019-11-27)
* `75`: ONNX: fix random_state whevever it is available when running benchmark (2019-11-27)

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
