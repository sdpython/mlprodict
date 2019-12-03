
.. _l-onnx-tutorial-optim:

Converters with options
=======================

Some converters have options to change the way
a specific operator is converted. The whole list
is described at :epkg:`Converters with options`.

.. contents::
    :local:

Option cdist for GaussianProcessRegressor
+++++++++++++++++++++++++++++++++++++++++

Notebooks :ref:`onnxpdistrst` shows how much slower
an :epkg:`ONNX` implementation of function
:epkg:`cdist`, from 3 to 10 times slower.
One way to optimize the converted model is to
create dedicated operators such as the one for function
:epkg:`cdist`. The first example shows how to
convert a :epkg:`GaussianProcessRegressor` into
standard :epkg:`ONNX` (see also @see cl CDist).

.. gdot::
    :script: DOT-SECTION

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ExpSineSquared
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = GaussianProcessRegressor(ExpSineSquared(), alpha=20.)
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train, dtype=numpy.float64)
    oinf = OnnxInference(model_def)
    print("DOT-SECTION", oinf.to_dot())

Now the new model with the operator `CDist`.

.. gdot::
    :script: DOT-SECTION

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ExpSineSquared
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = GaussianProcessRegressor(ExpSineSquared(), alpha=20.)
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train, dtype=numpy.float64,
                        options={GaussianProcessRegressor: {'optim': 'cdist'}})
    oinf = OnnxInference(model_def)
    print("DOT-SECTION", oinf.to_dot())

The only change is parameter *options*
set to ``options={GaussianProcessRegressor: {'optim': 'cdist'}}``.
It tells the conversion fonction that every every model
:epkg:`sklearn:gaussian_process:GaussianProcessRegressor`
must be converted with the option ``optim='cdist'``. The converter
of this model checks that that options and uses custom operator `CDist`
instead of its standard implementation based on operator
`Scan <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scan>`_.
Section :ref:`lpy-GaussianProcess` shows how much the gain
is depending on the number of observations for this example.
