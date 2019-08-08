
.. image:: https://travis-ci.org/sdpython/mlprodict.svg?branch=master
    :target: https://travis-ci.org/sdpython/mlprodict
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

.. image:: https://requires.io/github/sdpython/mlprodict/requirements.svg?branch=master
     :target: https://requires.io/github/sdpython/mlprodict/requirements/?branch=master
     :alt: Requirements Status

.. image:: https://codecov.io/github/sdpython/mlprodict/coverage.svg?branch=master
    :target: https://codecov.io/github/sdpython/mlprodict?branch=master

.. image:: http://img.shields.io/github/issues/sdpython/mlprodict.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/mlprodict/issues

.. image:: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/_images/nbcov.png
    :target: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/all_notebooks_coverage.html
    :alt: Notebook Coverage

.. image:: https://pepy.tech/badge/mlprodict
    :target: https://pypi.org/project/mlprodict/
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/mlprodict.svg
    :target: https://github.com/sdpython/mlprodict/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/mlprodict.svg
    :target: https://github.com/sdpython/mlprodict/
    :alt: Stars

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/sdpython/mlprodict/master?filepath=_doc%2Fnotebooks

.. _l-README:

mlprodict
=========

The packages explores ways to productionize machine learning predictions.
One approach uses *ONNX* and tries to implement
a runtime in python / numpy or wraps
`onnxruntime <https://github.com/Microsoft/onnxruntime>`_
into a single class. The package provides tools to compare
predictions, to benchmark models converted with
`sklearn-onnx <https://github.com/onnx/sklearn-onnx/tree/master/skl2onnx>`_.
The second approach consists in converting
a pipeline directly into C and is not much developed.

::

    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_iris
    from mlprodict.onnxrt import OnnxInference, measure_relative_difference
    import numpy

    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    lr = LinearRegression()
    lr.fit(X, y)

    # Predictions with scikit-learn.
    expected = lr.predict(X[:5])
    print(expected)

    # Conversion into ONNX.
    from mlprodict.onnxrt import to_onnx
    model_onnx = to_onnx(lr, X.astype(numpy.float32))

    # Predictions with onnxruntime
    oinf = OnnxInference(model_onnx, runtime='onnxruntime1')
    ypred = oinf.run({'X': X[:5]})
    print(ypred)

    # Measuring the maximum difference.
    print(measure_relative_difference(expected, ypred))

**Installation**

The project relies on *sklearn-onnx* which is in active
development. Continuous integration relies on a specific
branch of this project to benefit from the lastest changes:

::

    pip install git+https://github.com/xadupre/sklearn-onnx.git@jenkins

The project is currently in active development.
It is safer to install the package directly from
github:

::

    pip install git+https://github.com/sdpython/mlprodict.git

On Linux and Windows, the package must be compiled with
*openmp*. Full instructions to build the module and run
the documentation are described in `config.yml
<https://github.com/sdpython/mlprodict/blob/master/.circleci/config.yml>`_
for Linux. When this project becomes more stable,
it will changed to be using official releases.
Experiments with float64 are not supported with
``sklearn-onnx <= 1.5.0``.
The code is available at
`GitHub/mlprodict <https://github.com/sdpython/mlprodict/>`_
and has `online documentation <http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html>`_.
