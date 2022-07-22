
.. image:: https://github.com/sdpython/mlprodict/blob/master/_doc/sphinxdoc/source/phdoc_static/project_ico.png?raw=true
    :target: https://github.com/sdpython/mlprodict/

.. _l-README:

mlprodict
=========

.. image:: https://travis-ci.com/sdpython/mlprodict.svg?branch=master
    :target: https://app.travis-ci.com/github/sdpython/mlprodict/
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/g8chk1ufyk1m8uep?svg=true
    :target: https://ci.appveyor.com/project/sdpython/mlprodict
    :alt: Build Status Windows

.. image:: https://circleci.com/gh/sdpython/mlprodict/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/mlprodict/tree/master

.. image:: https://dev.azure.com/xavierdupre3/mlprodict/_apis/build/status/sdpython.mlprodict
    :target: https://dev.azure.com/xavierdupre3/mlprodict/

.. image:: https://badge.fury.io/py/mlprodict.svg
    :target: https://pypi.org/project/mlprodict/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://codecov.io/github/sdpython/mlprodict/coverage.svg?branch=master
    :target: https://codecov.io/github/sdpython/mlprodict?branch=master

.. image:: http://img.shields.io/github/issues/sdpython/mlprodict.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/mlprodict/issues

.. image:: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/_images/nbcov.png
    :target: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/all_notebooks_coverage.html
    :alt: Notebook Coverage

.. image:: https://pepy.tech/badge/mlprodict/month
    :target: https://pepy.tech/project/mlprodict/month
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/mlprodict.svg
    :target: https://github.com/sdpython/mlprodict/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/mlprodict.svg
    :target: https://github.com/sdpython/mlprodict/
    :alt: Stars

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/sdpython/mlprodict/master?filepath=_doc%2Fnotebooks

.. image:: https://img.shields.io/github/repo-size/sdpython/mlprodict
    :target: https://github.com/sdpython/mlprodict/
    :alt: size

*mlprodict* was initially started to help implementing converters
to *ONNX*. The main features is a python runtime for
*ONNX* (class `OnnxInference
<http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/onnxrt/onnx_inference.html>`_),
visualization tools
(see `Visualization
<http://www.xavierdupre.fr/app/mlprodict/helpsphinx/api/tools.html#visualization>`_),
and a `numpy API for ONNX
<http://www.xavierdupre.fr/app/mlprodict/helpsphinx/tutorial/numpy_api_onnx.html>`_).
The package also provides tools to compare
predictions, to benchmark models converted with
`sklearn-onnx <https://github.com/onnx/sklearn-onnx/tree/master/skl2onnx>`_.

::

    import numpy
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_iris
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.onnxrt.validate.validate_difference import measure_relative_difference
    from mlprodict import __max_supported_opset__, get_ir_version

    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    lr = LinearRegression()
    lr.fit(X, y)

    # Predictions with scikit-learn.
    expected = lr.predict(X[:5])
    print(expected)

    # Conversion into ONNX.
    from mlprodict.onnx_conv import to_onnx
    model_onnx = to_onnx(lr, X.astype(numpy.float32),
                         black_op={'LinearRegressor'},
                         target_opset=__max_supported_opset__)
    print("ONNX:", str(model_onnx)[:200] + "\n...")

    # Predictions with onnxruntime
    model_onnx.ir_version = get_ir_version(__max_supported_opset__)
    oinf = OnnxInference(model_onnx, runtime='onnxruntime1')
    ypred = oinf.run({'X': X[:5].astype(numpy.float32)})
    print("ONNX output:", ypred)

    # Measuring the maximum difference.
    print("max abs diff:", measure_relative_difference(expected, ypred['variable']))

    # And the python runtime
    oinf = OnnxInference(model_onnx, runtime='python')
    ypred = oinf.run({'X': X[:5].astype(numpy.float32)},
                     verbose=1, fLOG=print)
    print("ONNX output:", ypred)

**Installation**

Installation from *pip* should work unless you need the latest
development features.

::

    pip install mlprodict

The package includes a runtime for *ONNX*. That's why there
is a limited number of dependencies. However, some features
relies on *sklearn-onnx*, *onnxruntime*, *scikit-learn*.
They can be installed with the following instructions:

::

    pip install mlprodict[all]

The code is available at
`GitHub/mlprodict <https://github.com/sdpython/mlprodict/>`_
and has `online documentation <http://www.xavierdupre.fr/app/
mlprodict/helpsphinx/index.html>`_.
