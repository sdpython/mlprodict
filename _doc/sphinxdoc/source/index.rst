
.. |gitlogo| image:: _static/git_logo.png
             :height: 20

.. image:: https://github.com/sdpython/mlprodict/blob/master/_doc/sphinxdoc/source/phdoc_static/project_ico.png?raw=true
    :target: https://github.com/sdpython/mlprodict/

mlprodict
=========

**Links:** `github <https://github.com/sdpython/mlprodict/>`_,
`documentation <http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html>`_,
:ref:`l-README`,
:ref:`blog <ap-main-0>`

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

.. image:: https://pepy.tech/badge/mlprodict
    :target: https://pypi.org/project/mlprodict/
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/mlprodict.svg
    :target: https://github.com/sdpython/pyquickhelper/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/mlprodict.svg
    :target: https://github.com/sdpython/mlprodict/
    :alt: Stars

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/sdpython/mlprodict/master?filepath=_doc%2Fnotebooks

.. image:: https://img.shields.io/github/repo-size/sdpython/mlprodict
    :target: https://github.com/sdpython/mlprodict/
    :alt: size

.. toctree::
    :maxdepth: 1

    installation
    tutorial/index
    api/index
    onnx
    onnx_bench
    i_cmd
    i_ex
    i_index
    gyexamples/index
    all_notebooks
    HISTORY

*mlprodict* was initially started to help implementing converters
to :epkg:`ONNX`. The main feature is a python runtime for
:epkg:`ONNX`. It gives more feedback than :epkg:`onnxruntime`
when the execution fails.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning

    import numpy
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_iris
    from mlprodict.onnxrt import OnnxInference
    from mlprodict.onnxrt.validate.validate_difference import measure_relative_difference
    from mlprodict import get_ir_version

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
                         target_opset=15)
    print("ONNX:", str(model_onnx)[:200] + "\n...")

    # Predictions with onnxruntime
    model_onnx.ir_version = get_ir_version(15)
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

These predictions are obtained with the
following :epkg:`ONNX` graph.

.. gdot::
    :script: DOT-SECTION

    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_iris
    from mlprodict.onnxrt import OnnxInference
    import numpy

    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    lr = LinearRegression()
    lr.fit(X, y)

    from mlprodict.onnx_conv import to_onnx
    model_onnx = to_onnx(lr, X.astype(numpy.float32))
    oinf = OnnxInference(model_onnx)
    print("DOT-SECTION", oinf.to_dot())

Notebook :ref:`onnxvisualizationrst`
shows how to visualize an :epkg:`ONNX` pipeline.
The package also contains a collection of tools
to help converting code to ONNX. A short list of
them:

* **Python runtime for ONNX:**
  :class:`OnnxInference <mlprodict.onnxrt.onnx_infernce.OnnxInference>`,
  it is mostly used to check that an ONNX graph produces the expected output.
  If it fails, it fails within a python code and not inside C++ code.
  This class can also be used to call :epkg:`onnxruntime` by
  using ``runtime=='onnxruntime1'``. A last runtime
  ``runtime=='python_compiled'`` compiles a python function equivalent
  to code calling operator one by one. It makes easier to read the ONNX
  graph (see :ref:`l-onnx-tutorial`).
* **Intermediate results:**
  the python runtime may display all intermediate results,
  their shape if `verbosity == 1`, their value if `verbosity > 10`,
  see :ref:`l-onnx-tutorial`. This cannot be done with ``runtime=='onnxruntime1'``
  but it is still possible to get the intermediate results
  (see :meth:`OnnxInference.run <mlprodict.onnxrt.onnx_inference.OnnxInference.run>`).
  The class will build all subgraphs from the inputs to every intermediate
  results. If the graph has *N* operators, the cost of this will be
  :math:`O(N^2)`.
* **Extract a subpart of an ONNX graph:**
  hen an ONNX graph does not load, it is possible to modify, to extract
  some subpart to check a tiny part of it. Function
  :func:`select_model_inputs_outputs
  <mlprodict.onnx_tools.onnx_manipulations.select_model_inputs_outputs>`
  may be used to change the inputs and/or the outputs.
* **Change the opset**: function
  :func:`overwrite_opset
  <mlprodict.onnx_tools.onnx_manipulations.overwrite_opset>`
  overwrites the opset, it is used to check for which opset (ONNX version)
  a graph is valid. ...
* **Visualization in a notebook**: a magic command to display
  small ONNX graph in notebooks :ref:`onnxvisualizationrst`.
* **Text visualization for ONNX:** a way to visualize ONNX graph only
  with text :func:`onnx_text_plot <mlprodict.plotting.text_plot.onnx_text_plot>`.
* **Text visualization of TreeEnsemble:** a way to visualize the graph
  described by a on operator TreeEnsembleRegressor or TreeEnsembleClassifier,
  see :func:`onnx_text_plot <mlprodict.plotting.text_plot.onnx_text_plot_tree>`.
* **Export ONNX graph to numpy:** the numpy code produces the same
  results as the ONNX graph (see :func:`export2numpy
  <mlprodict.onnx_tools.onnx_export.export2numpy>`)
* **Export ONNX graph to ONNX API:** this produces a
  a code based on ONNX API which replicates the ONNX graph
  (see :func:`export2onnx
  <mlprodict.onnx_tools.onnx_export.export2onnx>`)
* **Export ONNX graph to** :epkg:`tf2onnx`: still a function which
  creates an ONNX graph but based on :epkg:`tf2onnx` API
  (see :func:`export2tf2onnx
  <mlprodict.onnx_tools.onnx_export.export2tf2onnx>`)
* **Xop API:** (ONNX operators API), see :ref:`l-xop-api`,
  most of the converting libraries uses :epkg:`onnx` to create ONNX graphs.
  The API is quite verbose and that is why most of them implement a second
  API wrapping the first one. They are not necessarily meant to be used
  by users to create ONNX graphs as they are specialized for the training
  framework they are developped for.
* **Numpy API for ONNX:** many functions doing computation are
  written with :epkg:`numpy` and converting them to ONNX may take
  quite some time for users not familiar with ONNX. This API implements
  many functions from :epkg:`numpy` with ONNX and allows the user
  to combine them. It is as if numpy function where exectued by an
  ONNX runtime: :ref:`l-numpy-api-for-onnx`.
* **Benchmark scikit-learn models converted into ONNX:** a simple function to
  benchmark ONNX against *scikit-learn* for a simple model:
  :ref:`l-example-onnx-benchmark`
* **Accelerate scikit-learn prediction:**,
  what if *transform* or *predict* is replaced by an implementation
  based on ONNX, or a numpy version of it, would it be faster?
  :ref:`l-Speedup-pca`
* **Profiling onnxruntime:** :epkg:`onnxruntime` can memorize the time
  spent in each operator. The following notebook shows how to retreive
  the results and display them :ref:`onnxprofileortrst`.

This package supports ONNX opsets to the latest opset stored
in `mlprodict.__max_supported_opset__` which is:

.. runpython::
    :showcode:

    import mlprodict
    print(mlprodict.__max_supported_opset__)

Any opset beyond that value is not supported and could fail.
That's for the main set of ONNX functions or domain.
Converters for :epkg:`scikit-learn` requires another domain,
`'ai.onnxml'` to implement tree. Latest supported options
are defined here:

.. runpython::
    :showcode:

    import pprint
    import mlprodict
    pprint.pprint(mlprodict.__max_supported_opsets__)

+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`l-modules`     |  :ref:`l-functions` | :ref:`l-classes`    | :ref:`l-methods`   | :ref:`l-staticmethods` | :ref:`l-properties`                            |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`modindex`      |  :ref:`l-EX2`       | :ref:`search`       | :ref:`l-license`   | :ref:`l-changes`       | :ref:`l-README`                                |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`genindex`      |  :ref:`l-FAQ2`      | :ref:`l-notebooks`  |                    | :ref:`l-statcode`      | `Unit Test Coverage <coverage/index.html>`_    |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
