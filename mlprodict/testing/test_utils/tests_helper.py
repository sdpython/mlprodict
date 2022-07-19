"""
@file
@brief Inspired from sklearn-onnx, handles two backends.
"""
import pickle
import os
import warnings
import traceback
import time
import sys
import numpy
import pandas
from sklearn.datasets import (
    make_classification, make_multilabel_classification,
    make_regression)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from .utils_backend import compare_backend
from .utils_backend_common import (
    extract_options, evaluate_condition, is_backend_enabled,
    OnnxBackendMissingNewOnnxOperatorException)


def _has_predict_proba(model):
    if hasattr(model, "voting") and model.voting == "hard":
        return False
    return hasattr(model, "predict_proba")


def _has_decision_function(model):
    if hasattr(model, "voting"):
        return False
    return hasattr(model, "decision_function")


def _has_transform_model(model):
    if hasattr(model, "voting"):
        return False
    return hasattr(model, "fit_transform") and hasattr(model, "score")


def fit_classification_model(model, n_classes, is_int=False,
                             pos_features=False, label_string=False,
                             random_state=42, is_bool=False,
                             n_features=20):
    """
    Fits a classification model.
    """
    X, y = make_classification(n_classes=n_classes, n_features=n_features,
                               n_samples=500,
                               random_state=random_state,
                               n_informative=7)
    if label_string:
        y = numpy.array(['cl%d' % cl for cl in y])
    X = X.astype(numpy.int64) if is_int or is_bool else X.astype(numpy.float32)
    if pos_features:
        X = numpy.abs(X)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def fit_multilabel_classification_model(model, n_classes=5, n_labels=2,
                                        n_samples=400, n_features=20,
                                        is_int=False):
    """
    Fits a classification model.
    """
    X, y = make_multilabel_classification(
        n_classes=n_classes, n_labels=n_labels, n_features=n_features,
        n_samples=n_samples, random_state=42)[:2]
    X = X.astype(numpy.int64) if is_int else X.astype(numpy.float32)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def fit_regression_model(model, is_int=False, n_targets=1, is_bool=False,
                         factor=1., n_features=10, n_samples=500,
                         n_informative=10):
    """
    Fits a regression model.
    """
    X, y = make_regression(n_features=n_features, n_samples=n_samples,
                           n_targets=n_targets, random_state=42,
                           n_informative=n_informative)[:2]
    y *= factor
    X = X.astype(numpy.int64) if is_int or is_bool else X.astype(numpy.float32)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def fit_classification_model_simple(model, n_classes, is_int=False,
                                    pos_features=False):
    """
    Fits a classification model.
    """
    X, y = make_classification(n_classes=n_classes, n_features=10,
                               n_samples=500, n_redundant=0,
                               n_repeated=0,
                               random_state=42, n_informative=9)
    X = X.astype(numpy.int64) if is_int else X.astype(numpy.float32)
    if pos_features:
        X = numpy.abs(X)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def _raw_score_binary_classification(model, X):
    scores = model.decision_function(X)
    if len(scores.shape) == 1:
        scores = scores.reshape(-1, 1)
    if len(scores.shape) != 2 or scores.shape[1] != 1:
        raise RuntimeError(  # pragma: no cover
            f"Unexpected shape {scores.shape} for a binary classifiation")
    return numpy.hstack([-scores, scores])


def _save_model_dump(model, folder, basename, names):
    if hasattr(model, "save"):  # pragma: no cover
        dest = os.path.join(folder, basename + ".model.keras")
        names.append(dest)
        model.save(dest)
    else:
        dest = os.path.join(folder, basename + ".model.pkl")
        names.append(dest)
        with open(dest, "wb") as f:
            try:
                pickle.dump(model, f)
            except AttributeError as e:  # pragma no cover
                print(
                    f"[dump_data_and_model] cannot pickle model '{dest}' due to {e}.")


def dump_data_and_model(  # pylint: disable=R0912
        data, model, onnx_model=None, basename="model", folder=None,
        inputs=None, backend=('python', 'onnxruntime'),
        context=None, allow_failure=None, methods=None,
        dump_error_log=None, benchmark=None, comparable_outputs=None,
        intermediate_steps=False, fail_evenif_notimplemented=False,
        verbose=False, classes=None, check_error=None, disable_optimisation=False):
    """
    Saves data with pickle, saves the model with pickle and *onnx*,
    runs and saves the predictions for the given model.
    This function is used to test a backend (runtime) for *onnx*.

    :param data: any kind of data
    :param model: any model
    :param onnx_model: *onnx* model or *None* to use an onnx converters to convert it
        only if the model accepts one float vector
    :param basename: three files are writen ``<basename>.data.pkl``,
        ``<basename>.model.pkl``, ``<basename>.model.onnx``
    :param folder: files are written in this folder,
        it is created if it does not exist, if *folder* is None,
        it looks first in environment variable ``ONNXTESTDUMP``,
        otherwise, it is placed into ``'temp_dump'``.
    :param inputs: standard type or specific one if specified, only used is
        parameter *onnx* is None
    :param backend: backend used to compare expected output and runtime output.
        Two options are currently supported: None for no test,
        `'onnxruntime'` to use module :epkg:`onnxruntime`,
        ``python`` to use the python runtiume.
    :param context: used if the model contains a custom operator such
        as a custom Keras function...
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :param dump_error_log: if True, dumps any error message in a file
        ``<basename>.err``, if it is None, it checks the environment
         variable ``ONNXTESTDUMPERROR``
    :param benchmark: if True, runs a benchmark and stores the results
        into a file ``<basename>.bench``, if None, it checks the environment
        variable ``ONNXTESTBENCHMARK``
    :param verbose: additional information
    :param methods: ONNX may produce one or several results, each of them
        is equivalent to the output of a method from the model class,
        this parameter defines which methods is equivalent to ONNX outputs.
        If not specified, it falls back into a default behaviour implemented
        for classifiers, regressors, clustering.
    :param comparable_outputs: compares only these outputs
    :param intermediate_steps: displays intermediate steps
        in case of an error
    :param fail_evenif_notimplemented: the test is considered as failing
        even if the error is due to onnxuntime missing the implementation
        of a new operator defiend in ONNX.
    :param classes: classes names
        (only for classifier, mandatory if option 'nocl' is used)
    :param check_error: do not raise an exception if the error message
        contains this text
    :param disable_optimisation: disable all optimisations *onnxruntime*
        could do
    :return: the created files

    Some convention for the name,
    *Bin* for a binary classifier, *Mcl* for a multiclass
    classifier, *Reg* for a regressor, *MRg* for a multi-regressor.
    The name can contain some flags. Expected outputs refer to the
    outputs computed with the original library, computed outputs
    refer to the outputs computed with a ONNX runtime.

    * ``-CannotLoad``: the model can be converted but the runtime
      cannot load it
    * ``-Dec3``: compares expected and computed outputs up to
      3 decimals (5 by default)
    * ``-Dec4``: compares expected and computed outputs up to
      4 decimals (5 by default)
    * ``-NoProb``: The original models computed probabilites for two classes
      *size=(N, 2)* but the runtime produces a vector of size *N*, the test
      will compare the second column to the column
    * ``-Out0``: only compares the first output on both sides
    * ``-Reshape``: merges all outputs into one single vector and resizes
      it before comparing
    * ``-SkipDim1``: before comparing expected and computed output,
      arrays with a shape like *(2, 1, 2)* becomes *(2, 2)*
    * ``-SklCol``: *scikit-learn* operator applies on a column and not a matrix

    If the *backend* is not None, the function either raises an exception
    if the comparison between the expected outputs and the backend outputs
    fails or it saves the backend output and adds it to the results.
    """
    # delayed import because too long
    from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType  # delayed

    runtime_test = dict(model=model, data=data)

    if folder is None:
        folder = os.environ.get("ONNXTESTDUMP", "temp_dump")
    if dump_error_log is None:
        dump_error_log = os.environ.get("ONNXTESTDUMPERROR", "0") in (
            "1", 1, "True", "true", True)
    if benchmark is None:
        benchmark = os.environ.get("ONNXTESTBENCHMARK", "0") in (
            "1", 1, "True", "true", True)
    if not os.path.exists(folder):
        os.makedirs(folder)

    lambda_original = None
    if isinstance(data, (numpy.ndarray, pandas.DataFrame)):
        dataone = data[:1].copy()
    else:
        dataone = data

    if methods is not None:
        prediction = []
        for method in methods:
            if callable(method):
                call = lambda X, model=model: method(
                    model, X)  # pragma: no cover
            else:
                try:
                    call = getattr(model, method)
                except AttributeError as e:  # pragma no cover
                    if method == 'decision_function_binary':
                        call = (
                            lambda X, model=model:
                                _raw_score_binary_classification(model, X))
                    else:
                        raise e
            if callable(call):
                prediction.append(call(data))
                # we only take the last one for benchmark
                lambda_original = lambda: call(dataone)
            else:
                raise RuntimeError(  # pragma: no cover
                    f"Method '{method}' is not callable.")
    else:
        if hasattr(model, "predict"):
            if _has_predict_proba(model):
                # Classifier
                prediction = [model.predict(data), model.predict_proba(data)]
                lambda_original = lambda: model.predict_proba(dataone)
            elif _has_decision_function(model):
                # Classifier without probabilities
                prediction = [model.predict(data),
                              model.decision_function(data)]
                lambda_original = (
                    lambda: model.decision_function(dataone))
            elif _has_transform_model(model):
                # clustering
                try:
                    prediction = [model.predict(data), model.transform(data)]
                    lambda_original = lambda: model.transform(dataone)
                except ValueError:
                    # 0.23 enforced type checking.
                    d64 = data.astype(numpy.float64)
                    prediction = [model.predict(d64), model.transform(d64)]
                    dataone64 = dataone.astype(numpy.float64)
                    lambda_original = lambda: model.transform(dataone64)
            else:
                # Regressor or VotingClassifier
                prediction = [model.predict(data)]
                lambda_original = lambda: model.predict(dataone)

        elif hasattr(model, "transform"):
            options = extract_options(basename)
            SklCol = options.get("SklCol", False)
            if SklCol:
                prediction = model.transform(data.ravel())  # pragma: no cover
                lambda_original = lambda: model.transform(
                    dataone.ravel())  # pragma: no cover
            else:
                prediction = model.transform(data)
                lambda_original = lambda: model.transform(dataone)
        else:
            raise TypeError(  # pragma: no cover
                f"Model has no predict or transform method: {type(model)}")

    runtime_test["expected"] = prediction

    names = []
    dest = os.path.join(folder, basename + ".expected.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(prediction, f)

    dest = os.path.join(folder, basename + ".data.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(data, f)

    _save_model_dump(model, folder, basename, names)

    if dump_error_log:  # pragma: no cover
        error_dump = os.path.join(folder, basename + ".err")

    if onnx_model is None:  # pragma: no cover
        array = numpy.array(data)
        if inputs is None:
            if array.dtype == numpy.float64:
                inputs = [("input", DoubleTensorType(list(array.shape)))]
            else:
                inputs = [("input", FloatTensorType(list(array.shape)))]
        onnx_model, _ = convert_model(model, basename, inputs)

    dest = os.path.join(folder, basename + ".model.onnx")
    names.append(dest)
    with open(dest, "wb") as f:
        f.write(onnx_model.SerializeToString())
    if verbose:  # pragma: no cover
        print(f"[dump_data_and_model] created '{dest}'.")

    runtime_test["onnx"] = dest

    # backend
    if backend is not None:
        if isinstance(backend, tuple):
            backend = list(backend)
        if not isinstance(backend, list):
            backend = [backend]
        for b in backend:
            if not is_backend_enabled(b):
                continue  # pragma: no cover
            if isinstance(allow_failure, str):
                allow = evaluate_condition(
                    b, allow_failure)  # pragma: no cover
            else:
                allow = allow_failure
            if allow is None and not check_error:
                output, lambda_onnx = compare_backend(
                    b, runtime_test, options=extract_options(basename),
                    context=context, verbose=verbose,
                    comparable_outputs=comparable_outputs,
                    intermediate_steps=intermediate_steps,
                    disable_optimisation=disable_optimisation,
                    classes=classes)
            elif check_error:
                try:
                    output, lambda_onnx = compare_backend(
                        b, runtime_test, options=extract_options(basename),
                        context=context, verbose=verbose,
                        comparable_outputs=comparable_outputs,
                        intermediate_steps=intermediate_steps,
                        disable_optimisation=disable_optimisation,
                        classes=classes)
                except Exception as e:  # pragma: no cover
                    if check_error in str(e):
                        warnings.warn(str(e))
                        continue
                    raise e
            else:
                try:
                    output, lambda_onnx = compare_backend(
                        b, runtime_test,
                        options=extract_options(basename),
                        context=context, verbose=verbose,
                        comparable_outputs=comparable_outputs,
                        intermediate_steps=intermediate_steps,
                        classes=classes)
                except OnnxBackendMissingNewOnnxOperatorException as e:  # pragma no cover
                    if fail_evenif_notimplemented:
                        raise e
                    warnings.warn(str(e))
                    continue
                except AssertionError as e:  # pragma no cover
                    if dump_error_log:
                        with open(error_dump, "w", encoding="utf-8") as f:
                            f.write(str(e) + "\n--------------\n")
                            traceback.print_exc(file=f)
                    if isinstance(allow, bool) and allow:
                        warnings.warn("Issue with '{0}' due to {1}".format(
                            basename,
                            str(e).replace("\n", " -- ")))
                        continue
                    raise e

            if output is not None:
                dest = os.path.join(folder,
                                    basename + f".backend.{b}.pkl")
                names.append(dest)
                with open(dest, "wb") as f:
                    pickle.dump(output, f)
                if (benchmark and lambda_onnx is not None and
                        lambda_original is not None):
                    # run a benchmark
                    obs = compute_benchmark({
                        "onnxrt": lambda_onnx,
                        "original": lambda_original
                    })
                    df = pandas.DataFrame(obs)
                    df["input_size"] = sys.getsizeof(dataone)
                    dest = os.path.join(folder, basename + ".bench")
                    df.to_csv(dest, index=False)

    return names


def convert_model(model, name, input_types):
    """
    Runs the appropriate conversion method.

    :param model: model, *scikit-learn*, *keras*,
         or *coremltools* object
    :param name: model name
    :param input_types: input types
    :return: *onnx* model
    """
    from skl2onnx import convert_sklearn  # delayed

    model, prefix = convert_sklearn(model, name, input_types), "Sklearn"
    if model is None:  # pragma: no cover
        raise RuntimeError(f"Unable to convert model of type '{type(model)}'.")
    return model, prefix


def dump_one_class_classification(
        model, suffix="", folder=None, allow_failure=None,
        comparable_outputs=None, verbose=False, benchmark=False,
        methods=None):
    """
    Trains and dumps a model for a One Class outlier problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    from skl2onnx.common.data_types import FloatTensorType  # delayed
    X = [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = [1, 1, 1]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "one_class",
                                       [("input", FloatTensorType([None, 2]))])
    dump_data_and_model(
        X, model, model_onnx, folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "One" + model.__class__.__name__ + suffix,
        verbose=verbose, comparable_outputs=comparable_outputs,
        benchmark=benchmark, methods=methods)


def dump_binary_classification(
        model, suffix="", folder=None, allow_failure=None,
        comparable_outputs=None, verbose=False, label_string=False,
        benchmark=False, methods=None, nrows=None):
    """
    Trains and dumps a model for a binary classification problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    from skl2onnx.common.data_types import FloatTensorType  # delayed
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    if label_string:
        y = ["A", "B", "A"]
    else:
        y = numpy.array([0, 1, 0], numpy.int64)
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "binary classifier",
                                       [("input", FloatTensorType([None, 2]))])
    if nrows == 2:
        for nr in range(X.shape[0] - 1):
            dump_data_and_model(
                X[nr: nr + 2], model, model_onnx, folder=folder, allow_failure=allow_failure,
                basename=prefix + "Bin" + model.__class__.__name__ + suffix,
                verbose=verbose, comparable_outputs=comparable_outputs, methods=methods)
    else:
        dump_data_and_model(
            X, model, model_onnx, folder=folder, allow_failure=allow_failure,
            basename=prefix + "Bin" + model.__class__.__name__ + suffix,
            verbose=verbose, comparable_outputs=comparable_outputs, methods=methods)

    X, y = make_classification(10, n_features=4, random_state=42)
    X = X[:, :2]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "binary classifier",
                                       [("input", FloatTensorType([None, 2]))])
    xt = X.astype(numpy.float32)
    if nrows is not None:
        xt = xt[:nrows]
    dump_data_and_model(
        xt, model, model_onnx,
        allow_failure=allow_failure, folder=folder,
        basename=prefix + "RndBin" + model.__class__.__name__ + suffix,
        verbose=verbose, comparable_outputs=comparable_outputs,
        benchmark=benchmark, methods=methods)


def dump_multiple_classification(
        model, suffix="", folder=None, allow_failure=None, verbose=False,
        label_string=False, first_class=0, comparable_outputs=None,
        benchmark=False, methods=None):
    """
    Trains and dumps a model for a binary classification problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    from skl2onnx.common.data_types import FloatTensorType  # delayed
    X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
    X = numpy.array(X, dtype=numpy.float32)
    y = [0, 1, 2, 1, 1, 2]
    y = [i + first_class for i in y]
    if label_string:
        y = ["l%d" % i for i in y]
    model.fit(X, y)
    if verbose:  # pragma: no cover
        print(
            f"[dump_multiple_classification] model '{model.__class__.__name__}'")
    model_onnx, prefix = convert_model(model, "multi-class classifier",
                                       [("input", FloatTensorType([None, 2]))])
    if verbose:  # pragma: no cover
        print("[dump_multiple_classification] model was converted")
    dump_data_and_model(
        X.astype(numpy.float32), model, model_onnx, folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "Mcl" + model.__class__.__name__ + suffix,
        verbose=verbose, comparable_outputs=comparable_outputs,
        methods=methods)

    X, y = make_classification(40, n_features=4, random_state=42,
                               n_classes=3, n_clusters_per_class=1)
    X = X[:, :2]
    model.fit(X, y)
    if verbose:  # pragma: no cover
        print(
            f"[dump_multiple_classification] model '{model.__class__.__name__}'")
    model_onnx, prefix = convert_model(model, "multi-class classifier",
                                       [("input", FloatTensorType([None, 2]))])
    if verbose:  # pragma: no cover
        print("[dump_multiple_classification] model was converted")
    dump_data_and_model(
        X[:10].astype(numpy.float32), model, model_onnx, folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "RndMcl" + model.__class__.__name__ + suffix,
        verbose=verbose, comparable_outputs=comparable_outputs,
        benchmark=benchmark, methods=methods)


def dump_multilabel_classification(
        model, suffix="", folder=None, allow_failure=None, verbose=False,
        label_string=False, first_class=0, comparable_outputs=None,
        benchmark=False, backend=('python', 'onnxruntime')):
    """
    Trains and dumps a model for a binary classification problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    from skl2onnx.common.data_types import FloatTensorType  # delayed
    X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
    X = numpy.array(X, dtype=numpy.float32)
    if label_string:
        y = [["l0"], ["l1"], ["l2"], ["l0", "l1"], ["l1"], ["l2"]]
    else:
        y = [[0 + first_class], [1 + first_class], [2 + first_class],
             [0 + first_class, 1 + first_class],
             [1 + first_class], [2 + first_class]]
    y = MultiLabelBinarizer().fit_transform(y)
    model.fit(X, y)
    if verbose:  # pragma: no cover
        print(
            f"[make_multilabel_classification] model '{model.__class__.__name__}'")
    model_onnx, prefix = convert_model(model, "multi-label-classifier",
                                       [("input", FloatTensorType([None, 2]))])
    if verbose:  # pragma: no cover
        print("[make_multilabel_classification] model was converted")
    dump_data_and_model(
        X.astype(numpy.float32), model, model_onnx, folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "Mcl" + model.__class__.__name__ + suffix,
        verbose=verbose, comparable_outputs=comparable_outputs,
        backend=backend)

    X, y = make_multilabel_classification(  # pylint: disable=W0632
        40, n_features=4, random_state=42, n_classes=3)
    X = X[:, :2]
    model.fit(X, y)
    if verbose:  # pragma: no cover
        print(
            f"[make_multilabel_classification] model '{model.__class__.__name__}'")
    model_onnx, prefix = convert_model(model, "multi-class classifier",
                                       [("input", FloatTensorType([None, 2]))])
    if verbose:  # pragma: no cover
        print("[make_multilabel_classification] model was converted")
    dump_data_and_model(
        X[:10].astype(numpy.float32), model, model_onnx, folder=folder,
        allow_failure=allow_failure,
        basename=prefix + "RndMla" + model.__class__.__name__ + suffix,
        verbose=verbose, comparable_outputs=comparable_outputs,
        benchmark=benchmark, backend=backend)


def dump_multiple_regression(
        model, suffix="", folder=None, allow_failure=None,
        comparable_outputs=None, verbose=False, benchmark=False):
    """
    Trains and dumps a model for a multi regression problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    from skl2onnx.common.data_types import FloatTensorType  # delayed
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = numpy.array([[100, 50], [100, 49], [100, 99]], dtype=numpy.float32)
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "multi-regressor",
                                       [("input", FloatTensorType([None, 2]))])
    dump_data_and_model(
        X, model, model_onnx, folder=folder, allow_failure=allow_failure,
        basename=prefix + "MRg" + model.__class__.__name__ + suffix,
        verbose=verbose, comparable_outputs=comparable_outputs,
        benchmark=benchmark)


def dump_single_regression(model, suffix="", folder=None, allow_failure=None,
                           comparable_outputs=None, benchmark=False):
    """
    Trains and dumps a model for a regression problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    from skl2onnx.common.data_types import FloatTensorType  # delayed
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = numpy.array([100, -10, 50], dtype=numpy.float32)
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, "single regressor",
                                       [("input", FloatTensorType([None, 2]))])
    dump_data_and_model(
        X, model, model_onnx, folder=folder, allow_failure=allow_failure,
        basename=prefix + "Reg" + model.__class__.__name__ + suffix,
        comparable_outputs=comparable_outputs)


def timeit_repeat(fct, number, repeat):
    """
    Returns a series of *repeat* time measures for
    *number* executions of *code* assuming *fct*
    is a function.
    """
    res = []
    for _ in range(0, repeat):
        t1 = time.perf_counter()
        for __ in range(0, number):
            fct()
        t2 = time.perf_counter()
        res.append(t2 - t1)
    return res


def timeexec(fct, number, repeat):
    """
    Measures the time for a given expression.

    :param fct: function to measure (as a string)
    :param number: number of time to run the expression
        (and then divide by this number to get an average)
    :param repeat: number of times to repeat the computation
        of the above average
    :return: dictionary
    """
    rep = timeit_repeat(fct, number=number, repeat=repeat)
    ave = sum(rep) / (number * repeat)
    std = (sum((x / number - ave)**2 for x in rep) / repeat)**0.5
    fir = rep[0] / number
    fir3 = sum(rep[:3]) / (3 * number)
    las3 = sum(rep[-3:]) / (3 * number)
    rep.sort()
    mini = rep[len(rep) // 20] / number
    maxi = rep[-len(rep) // 20] / number
    return dict(average=ave, deviation=std, first=fir, first3=fir3,
                last3=las3, repeat=repeat, min5=mini, max5=maxi, run=number)


def compute_benchmark(fcts, number=10, repeat=100):
    """
    Compares the processing time several functions.

    :param fcts: dictionary ``{'name': fct}``
    :param number: number of time to run the expression
        (and then divide by this number to get an average)
    :param repeat: number of times to repeat the computation
        of the above average
    :return: list of [{'name': name, 'time': ...}]
    """
    obs = []
    for name, fct in fcts.items():
        res = timeexec(fct, number=number, repeat=repeat)
        res["name"] = name
        obs.append(res)
    return obs


def binary_array_to_string(mat):
    """
    Used to compare decision path.
    """
    if not isinstance(mat, numpy.ndarray):
        raise NotImplementedError(  # pragma: no cover
            "Not implemented for other types than arrays.")
    if len(mat.shape) != 2:
        raise NotImplementedError(  # pragma: no cover
            "Not implemented for other arrays than matrices.")
    res = [[str(i) for i in row] for row in mat.tolist()]
    return [''.join(row) for row in res]
