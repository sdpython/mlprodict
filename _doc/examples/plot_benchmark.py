"""
Sample to benchmark a model against ONNX
========================================


This benchmark compares the predictions of a RandomForestRegressor
when the parallelization is disabled. The benchmark depends on
four parameters:
* number of estimators (-e)
* number of observations in the batch (-n)
* number of features (-f)
* number of repetitions (-r)
* use assume_finite (-a)
* onnx comparison (-o)
* number of jobs (-j)
Option `-o` uses module *onnxruntime* and *mlprodict*
which provide two C++ implementation of the random forest
predict method. It requires a conversion of the model
into *onnx* done by modules *sklearn-onnx* and *onnx*.
The first execution fails after saving a trained model
to make sure the training is not part of the benchmark.
*py-psy* can be run using the following command line:
::
    py-spy record --native --function --rate=10 -o profile.svg --
    python bench_random_forest_parallel_predict.py -e 100 -n 1 -f 10 -r 1000
"""
import argparse
import time
import pickle
import os
import numpy as np
from sklearn import config_context
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression



def build_model(m, e, n, f, o, j, opts):
    filename = "%s-e%d-n%d-f%d-onnx%d-nj%d-opts%s.pkl" % (
        m, e, n, f, o, j, opts)
    if os.path.exists(filename):
        print("restores %s e=%d n=%d f=%d onnx=%d n_jobs=%d options=%s" % (
            m, e, n, f, o, j, opts))
        with open(filename, "rb") as f:
            return pickle.load(f)

    print("training %s e=%d n=%d f=%d onnx=%d n_jobs=%d opts=%s" % (
        m, e, n, f, o, j, opts))
    if m == 'RandomForestRegressor':
        rf = RandomForestRegressor(n_estimators=e, random_state=1, n_jobs=j)
    elif m == 'GradientBoostingRegressor':
        rf = GradientBoostingRegressor(n_estimators=e, random_state=1, n_jobs=j)
    elif m == 'LinearRegressor':
        rf = LinearRegressor(random_state=1)
    elif m == 'LogisticRegression':
        rf = LogisticRegression(random_state=1)
    else:
        raise ValueError("Unexpected %r." % m)

    if hasattr(rf, 'predict_proba'):
        nt = 10000
        X, y = make_classification(
            n_samples=nt + n, n_features=f, n_informative=f // 2,
            random_state=1, n_classes=2)
        X_train, X_test = X[:nt], X[nt:]
        y_train = y[:nt]
    else:
        nt = 10000
        X, y = make_regression(
            n_samples=nt + n, n_features=f, n_informative=f // 2,
            n_targets=1, random_state=1)
        X_train, X_test = X[:nt], X[nt:]
        y_train = y[:nt]

    rf.fit(X_train, y_train)

    data = dict(model=rf, data=X_test.astype(np.float32))
    if o:
        # compares with onnx
        print("convert to onnx")
        from skl2onnx import to_onnx
        if 'cdist' in opts:
            options = {id(rf): {'optim': 'cdist'}}
        elif 'nozipmap' in opts:
            options = {id(rf): {'zipmap': False}}
        else:
            options = None
        model_onnx = to_onnx(rf, X_train[:1].astype(np.float32), options=options)
        buffer_onnx = model_onnx.SerializeToString()
        data['onnx'] = buffer_onnx

    print("saving to '%s'" % filename)
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    return data


def _run_predict(model, X, repeat):
    for r in range(repeat):
        model.predict(X)


def skl_predict(model, X, repeat):
    _run_predict(model, X, repeat)


def ort_predict(sess, X, repeat):
    for r in range(repeat):
        sess.run(None, {'X': X})


def pyrt_predict(sess, X, repeat):
    for r in range(repeat):
        sess.run({'X': X})


def benchmark(model, model_onnx, X, repeat):
    begin = time.perf_counter()
    skl_predict(model, X, repeat)
    end1 = time.perf_counter()
    r = repeat
    e = len(getattr(model, 'estimators_', [1]))
    n = X.shape[0]
    f = X.shape[1]
    print("scikit-learn predict: r=%d e=%d n=%d f=%d time=%f" % (
        r, e, n, f, end1 - begin))

    if model_onnx is not None:
        from onnxruntime import InferenceSession
        from mlprodict.onnxrt import OnnxInference
        sess = InferenceSession(model_onnx)
        oinf = OnnxInference(model_onnx, runtime='python_compiled')

        begin = time.perf_counter()
        ort_predict(sess, X, repeat)
        end = time.perf_counter()
        print(" onnxruntime predict: r=%d e=%d n=%d f=%d time=%f" % (
            r, e, n, f, end - begin))

        begin = time.perf_counter()
        pyrt_predict(oinf, X, repeat)
        end = time.perf_counter()
        print("   mlprodict_predict: r=%d e=%d n=%d f=%d time=%f" % (
            r, e, n, f, end - begin))


def main(m="LogisticRegression", e=100, n=10000,
         f=10, r=1000, a=True, o=True, j=2, opts=""):
    """
    Builds a model and benchmarks the model converted into ONNX.
    
    :param m: model name or experiment
    :param e: number of estimators or trees
    :param n: number of rows
    :param f: number of features
    :param r: number of repetitions
    :param a: assume finite or not
    :param o: compares to ONNX
    :param j: n_jobs
    :param opts: options
    """
    model_data = build_model(m, e, n, f, o, j, opts)    

    if a:
        with config_context(assume_finite=True):
            benchmark(model_data['model'], model_data.get('onnx', None),
                      model_data['data'], r)
    else:
        benchmark(model_data['model'], model_data.get('onnx', None),
                  model_data['data'], r)

# Use py-spy: `py-spy record --native -r 10 -o plot_benchmark.svg -- python plot_benchmark.py`
main()
    