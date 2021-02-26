
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
standard :epkg:`ONNX` (see also :class:`CDist
<mlprodict.onnxrt.ops_cpu.op_cdist.CDist>`).

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

    model_def = to_onnx(clr, X_train)
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

    model_def = to_onnx(clr, X_train,
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

Other model supported cdist
+++++++++++++++++++++++++++

Pairwise distances are also is all nearest neighbours models.
That same *cdist* option is also supported for these models.

Option *zipmap* for classifiers
+++++++++++++++++++++++++++++++

By default, the library *sklearn-onnx* produces a list
of dictionaries ``{label: prediction}`` but this data structure
takes a significant time to be build. The converted
model can stick to matrices by removing operator *ZipMap*.
This is done by using option ``{'zipmap': False}``.

.. gdot::
    :script: DOT-SECTION

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = LogisticRegression()
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train,
                        options={LogisticRegression: {'zipmap': False}})
    oinf = OnnxInference(model_def)
    print("DOT-SECTION", oinf.to_dot())

Option *raw_scores* for classifiers
+++++++++++++++++++++++++++++++++++

By default, the library *sklearn-onnx* produces probabilities
whenever it is possible for a classifier. Raw scores can usually
be still obtained by using option ``{'raw_scores': True}``.

.. gdot::
    :script: DOT-SECTION

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = LogisticRegression()
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train,
                        options={LogisticRegression: {
                            'zipmap': False, 'raw_scores': True}})
    oinf = OnnxInference(model_def)
    print("DOT-SECTION", oinf.to_dot())

Pickability and Pipeline
++++++++++++++++++++++++

The proposed way to specify options is not always pickable.
Function ``id(model)`` depends on the execution and map an option
to one class may be not enough to customize the conversion.
However, it is possible to specify an option the same way
parameters are referenced in a *scikit-learn* pipeline
with method `get_params <https://scikit-learn.org/stable/modules/generated/
sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.get_params>`_.
Following syntax are supported:

::

    pipe = Pipeline([('pca', PCA()), ('classifier', LogisticRegression())])

    options = {'classifier': {'zipmap': False}}

Or

::

    options = {'classifier__zipmap': False}

Options applied to one model, not a pipeline as the converter
replaces the pipeline structure by a single onnx graph.
Following that rule, option *zipmap* would not have any impact
if applied to a pipeline and to the last step of the pipeline.
However, because there is no ambiguity about what the conversion
should be, for options *zipmap* and *nocl*, the following
options would have the same effect:

::

    pipe = Pipeline([('pca', PCA()), ('classifier', LogisticRegression())])

    options = {id(pipe.steps[-1][1]): {'zipmap': False}}
    options = {id(pipe): {'zipmap': False}}
    options = {'classifier': {'zipmap': False}}
    options = {'classifier__zipmap': False}
