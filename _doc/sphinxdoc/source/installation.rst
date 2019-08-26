
Installation
============

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

.. toctree::

    README
