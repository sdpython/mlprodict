
Installation
============

Installation from *pip* should work unless you need the latest
development features.

::

    pip install mlprodict

The package includes a runtime for *onnx*. That's why there
is a limited number of dependencies. However, some features
relies on *sklearn-onnx*, *onnxruntime*, *scikit-learn*.
They can be installed with the following instructions:

::

    pip install mlprodict[all]

Some functions used in that package may rely on features
implemented in PR still pending. In that case, you should
install *sklearn-onnx* from:

::

    pip install git+https://github.com/xadupre/sklearn-onnx.git@jenkins

If needed, the development version should be directy installed
from github:

::

    pip install git+https://github.com/sdpython/mlprodict.git

On Linux and Windows, the package must be compiled with
*openmp*. Full instructions to build the module and run
the documentation are described in `config.yml
<https://github.com/sdpython/mlprodict/blob/master/.circleci/config.yml>`_
for Linux. When this project becomes more stable,
it will changed to be using official releases.

.. toctree::

    README
