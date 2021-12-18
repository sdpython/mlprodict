
.. _l-onnx-tutorial-benchmark:

Benchmarks
==========

.. contents::
    :local:

.. _l-benchmark-onnxruntime-skl-regular:

Validates a runtime against scikit-learn
++++++++++++++++++++++++++++++++++++++++

This reuse the example :ref:`l-example-onnx-benchmark`.
The goal is to compare different implementation of an operator.

**Step 1: create a virtual environment**

::

    python -m virtualenv baseline

All other steps are executed within the local environment.

**Step 2: install the packages**

::

    pip install numpy scikit-learn onnx onnxruntime skl2onnx pyquickhelper matplotlib mlprodict threadpoolctl lightgbm xgboost

**Step 3: run the benchmark**

::

    export model="RandomForestRegressor"
    python -m mlprodict validate_runtime --n_features 4,50 -nu 2 -re 2 -o 11 -op 11 -v 1 --out_raw data$model.csv --out_summary summary$model.csv -b 1 --dump_folder dump_errors --runtime python_compiled,onnxruntime1 --models $model --out_graph bench_png$model --dtype 32

A full example is available on the following page.

.. toctree::
    :maxdepth: 1

    benchmarkorts

Compares two different onnxruntime
++++++++++++++++++++++++++++++++++

The benchmark is done using :epkg:`asv`.
This section assumes
there exist a local pypi server with the different tested
versions. It can be done with module :epkg:`pypiserver`.

**Step 0: pypiserver**

Copy all the necessary packages in a local folder
and starts a local :epkg:`pypiserver`.

::

    python -m pypiserver -u -p 8067 --disable-fallback local_pypi_server

**Step 1: virtual environment**

::

    python -m virtualenv pyenv

Everything else is done from the virtual environment.

::

    pip install numpy scikit-learn matplotlib pandas cython pybind11 lightgbm xgboost virtualenv pyquickhelper

The module :epkg:`asv` may work, otherwise a modified version
is necessary:

::

    pip install git+https://github.com/sdpython/asv.git@jenkins

And the local packages:

::

    pip install --upgrade --index http://localhost:8067/simple/ skl2onnx onnxconverter_common mlprodict onnxruntime onnx --extra-index-url=https://pypi.python.org/simple/

**Step 2: create the benchmark**

Versions must be verified.

::

    python -m mlprodict asv_bench  --location . --models "RandomForestRegressor" --build "build" -dt 32 -ma "{\"onnxruntime\":[\"1.1.2\", \"http://localhost:8067/simple/\"],\"onnx\":[\"1.6.0\"],\"scikit-learn\":[\"0.22.2.post1\"]}" -n 4 -o 11 -op 11 -r scikit-learn,python_compiled,onnxruntime1

**Step 3: run the benchmak**

::

    python -m asv run --show-stderr --config=asv.conf.json

**Step 4: publish the bechmark**

::

    python -m asv publish -o html --config=asv.conf.json

**Step 5: export into ASV format**

::

    python -m mlprodict asv2csv -f . -o asv_benchmark.csv -b skl -c asv.conf.json
