
.. _l-numpy-api-for-onnx:

Create ONNX graphs with an API similar to numpy
===============================================

Many people came accross the task of converting a pipeline
including a custom preprocessing embedded into a
:epkg:`sklearn:preprocessing:FunctionTransformer`.
:epkg:`sklearn-onnx` implements many converters. Their task
is to create an ONNX graph for every :epkg:`scikit-learn`
model included in a pipeline. Every converter is a new implementation
of methods `predict`, `predict_proba` or `transform` with
:epkg:`ONNX Operators`. But that does not include custom function.
Writing a converter can be quite verbose and requires to know
the :epkg:`ONNX Operators`, similar to :epkg:`numpy` but not
the same.

The goal here is to make it easier for users and have their custom
function converted in ONNX.
Everybody playing with :epkg:`scikit-learn` knows :epkg:`numpy`
then it should be possible to write a function using :epkg:`numpy`
and automatically have it converted into :epkg:`ONNX`.

.. contents::
    :local:

Available notebooks:

* :ref:`numpyapionnxrst`

Principle
+++++++++

The user writes a function using :epkg:`numpy` function but
behind the scene, it uses an :epkg:`ONNX` runtime to execute
the function. To do that, this package reimplements many
:epkg:`numpy` functions using :epkg:`ONNX Operators`. It looks
like :epkg:`numpy` but it uses :epkg:`ONNX`.
Following example shows how to replace *numpy* by *ONNX*.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray

    # The numpy function
    def log_1(x):
        return np.log(x + 1)

    # The ONNX function
    @onnxnumpy_default
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[Any, np.float32]:
        return npnx.log(x + np.float32(1))

    x = np.random.rand(2, 3).astype(np.float32)

    print('numpy')
    print(log_1(x))

    print('onnx')
    print(onnx_log_1(x))

ONNX runtimes are usually more strict about types than :epkg:`numpy`
(see :epkg:`onnxruntime`).
A function must be implemented for the same input type
and there is not implicit cast. There are three important elements
in this example:

* Decorator :func:`onnxnumpy_default <mlprodict.npy.onnx_numpy_wrapper.onnxnumpy_default>`:
  it parses the annotations, creates the ONNX graph and initialize a runtime with it.
* Annotation: every input and output types must be specified. They are :class:`NDArray
  <mlprodict.npy.onnx_numpy_annotation.NDArray>`, shape can be left undefined by element
  type must be precised.
* Types: `1` is different than `np.float32(1)`, the right type must be used.

`onnx_log_1` is not a function but an instance of class
:class:`wrapper_onnxnumpy <mlprodict.npy.onnx_numpy_wrapper.wrapper_onnxnumpy>`.
This class implements method `__call__` to behave like a function
and holds an attribute of type
:class:`OnnxNumpyCompiler <mlprodict.npy.onnx_numpy_compiler.OnnxNumpyCompiler>`.
This class contains an ONNX graph and a instance of a runtime.
The following lines lists some usefull attributes.

* `onnx_log_1`: :class:`wrapper_onnxnumpy <mlprodict.npy.onnx_numpy_wrapper.wrapper_onnxnumpy>`
* `onnx_log_1.compiled`: :class:`OnnxNumpyCompiler <mlprodict.npy.onnx_numpy_compiler.OnnxNumpyCompiler>`
* `onnx_log_1.compiled.onnx_`: ONNX graph
* `onnx_log_1.compiled.rt_fct_.rt`: runtime, by default
  :class:`OnnxInference <mlprodict.onnxrt.onnx_inference.OnnxInference>`

The ONNX graph `onnx_log_1.compiled.onnx_` looks like this:

.. gdot::
    :script: DOT-SECTION
    :process:

    from typing import Any
    import numpy as np
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray

    # The ONNX function
    @onnxnumpy_default
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[Any, np.float32]:
        return npnx.log(x + np.float32(1))

    onx = onnx_log_1.compiled.onnx_
    print(onx)

    oinf = onnx_log_1.compiled.rt_fct_.rt
    print("DOT-SECTION", oinf.to_dot())

There is a fundamental different between :epkg:`numpy` and
:epkg:`ONNX`. :epkg:`numpy` allows inplace modifications.
The simple instruction ``m[:, 0] = 1`` modifies an entire column
of an existing array. :epkg:`ONNX` does not allow that, even if the
same operator can be achieved, the result is a new array.
See section :ref:`l-inplace-modification-onnx` for more
details.

Available functions
+++++++++++++++++++

This tool does not implement every function of :epkg:`numpy`.
This a work in progress. The list of supported functions is
available at :ref:`f-numpyonnximpl`.

Common operators `+`, `-`, `/`, `*`,  `**`, `%`, `[]` are
supported as well. They are implemented by class
:class:`OnnxVar <mlprodict.npy.onnx_variable.OnnxVar>`.
This class also implements methods such as `astype` or
properties such as `shape`, `size`, `T`.

FunctionTransformer
+++++++++++++++++++

Now onnx was used to implement a custom function,
it needs to used by a :epkg:`sklearn:preprocessing:FunctionTransformer`.
One instance is added in a pipeline trained on the Iris dataset.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    @onnxnumpy_default
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
        return npnx.log(x + np.float32(1))

    data = load_iris()
    X, y = data.data.astype(np.float32), data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe = make_pipeline(
                FunctionTransformer(onnx_log_1),
                StandardScaler(),
                LogisticRegression())
    pipe.fit(X_train, y_train)
    print(pipe.predict_proba(X_test[:2]))

    onx = to_onnx(pipe, X_train[:1], rewrite_ops=True,
                  options={LogisticRegression: {'zipmap': False}})
    oinf = OnnxInference(onx)
    print(oinf.run({'X': X_test[:2]})['probabilities'])

*ONNX* is still more strict than *numpy*. Some elements
must be added every time this is used:

* The custom function signature is using *float32*,
  training and testing data are cast in *float32*.
* The shape of `onnx_log_1` return was changed into
  `NDArray[(None, None), np.float32]`. Otherwise the converter
  for *StandardScaler* raised an exception (see
  :ref:`l-npy-shape-mismatch`).
* Method :func:`to_onnx <mlprodict.onnx_conv.convert.to_onnx>`
  is called with parameter `rewrite_ops=True`. This parameter
  tells the function to overwrite the converter for
  *FunctionTransformer* by a new one which supports custom
  functions implemented with this API (see
  :ref:`l-npy-missing-converter`).

More options
++++++++++++

Use onnxruntime as ONNX runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the ONNX graph is executed by the Python runtime
implemented in this module (see :ref:`l-onnx-python-runtime`).
It is a mix of :epkg:`numpy` and C++ implementations and it does
not require any new dependency. However, it is possible to use
a different one like :epkg:`onnxruntime` which has an implementation
for more :epkg:`ONNX Operators`. The only change is a wrapper
with arguments :class:`onnxnumpy_np
<mlprodict.npy.onnx_numpy_wrapper.onnxnumpy_np>`:
`@onnxnumpy_np(runtime='onnxruntime')`.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from onnxruntime import InferenceSession
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_np, NDArray
    from mlprodict.onnx_conv import to_onnx

    @onnxnumpy_np(runtime='onnxruntime')
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
        return npnx.log(x + np.float32(1))

    data = load_iris()
    X, y = data.data.astype(np.float32), data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe = make_pipeline(
                FunctionTransformer(onnx_log_1),
                StandardScaler(),
                LogisticRegression())
    pipe.fit(X_train, y_train)
    print(pipe.predict_proba(X_test[:2]))

    onx = to_onnx(pipe, X_train[:1], rewrite_ops=True,
                  options={LogisticRegression: {'zipmap': False}})

    oinf = InferenceSession(onx.SerializeToString())
    print(oinf.run(None, {'X': X_test[:2]})[1])

Use a specific ONNX opset
^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the ONNX graph generated by the wrapper is using
the latest version of ONNX but it is possible to use an older one
if the involved runtime does not implement the latest version.
The desired opset must be specified in two places,
the first time as an argument of `onnxnumpy_np`, the second time
as an argument of `to_onnx`.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from onnxruntime import InferenceSession
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_np, NDArray
    from mlprodict.onnx_conv import to_onnx

    target_opset = 11

    @onnxnumpy_np(op_version=target_opset)  # first place
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
        return npnx.log(x + np.float32(1))

    data = load_iris()
    X, y = data.data.astype(np.float32), data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe = make_pipeline(
                FunctionTransformer(onnx_log_1),
                StandardScaler(),
                LogisticRegression())
    pipe.fit(X_train, y_train)
    print(pipe.predict_proba(X_test[:2]))

    onx = to_onnx(pipe, X_train[:1], rewrite_ops=True,
                  options={LogisticRegression: {'zipmap': False}},
                  target_opset=target_opset)  # second place

    oinf = InferenceSession(onx.SerializeToString())
    print(oinf.run(None, {'X': X_test[:2]})[1])

Same implementation for float32 and float64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only one input type is allowed by default but there is a way
to define a function supporting more than one type with
:class:`NDArrayType <mlprodict.npy.onnx_numpy_annotation.NDArrayType>`.
When calling function `onnx_log_1`, inputs are detected,
an ONNX graph is generated and executed. Next time the same function
is called, if the input types are the same as before, it reuses the same
ONNX graph and same runtime. Otherwise, it generates a new
ONNX graph taking this new type as input. The expression
`x.dtype` returns the type of this input in order to cast
the constant `1` into the right type before being used by
another operator.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
    :process:

    import numpy as np
    from onnxruntime import InferenceSession
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_np, NDArray
    from mlprodict.npy.onnx_numpy_annotation import NDArrayType
    from mlprodict.onnx_conv import to_onnx

    @onnxnumpy_np(signature=NDArrayType('floats'), runtime='onnxruntime')
    def onnx_log_1(x):
        return npnx.log(x + x.dtype(1))

    x = np.random.rand(2, 3)
    y = onnx_log_1(x.astype(np.float32))
    print(y.dtype, y)

    y = onnx_log_1(x.astype(np.float64))
    print(y.dtype, y)

There are more options to it. Many of them are used in
:ref:`f-numpyonnxpyrt`. It is possible to add arguments
with default values or undefined number of inputs. One
important detail though, a different value for an argument
(not an input) means the ONNX graph has to be different because
this value is stored in the graph instead of being an input.
Everytime an input type or an argument is different, a new ONNX
graph is generated and executed.

.. _l-inplace-modification-onnx:

How to convert inplace modifications
++++++++++++++++++++++++++++++++++++

As mentioned earlier, there is no way to modify a tensor inplace.
Every modification implies a copy. A modification can be done
by creating a new tensor concatenated from other tensors or by using
operators :epkg:`Op:ScatterElements` or :epkg:`Op:ScatterND`.
Instruction ``v[5] = 3.5`` is correct with numpy. Class :class:`OnnxVar
<mlprodict.npy.onnx_variable.OnnxVar>` replaces that instruction
with operator :epkg:`Op:ScatterElements`.

Operator `[] (__setitem__)` must return the instance itself (`self`).
That's why the design is different from the other methods. Instead of
returning a new instance of :class:`OnnxVar
<mlprodict.npy.onnx_variable.OnnxVar>`, it replaces the only input.
However, that require the operator `[]` to follow a copy.
``v[5] = 3.5`` may not be valid but ``v = v.copy(); v[5] = 3.5`` always is.
Current implementation only supports one dimensional tensor.
Operators :epkg:`Op:ScatterElements` or :epkg:`Op:ScatterND` are not
really meant to change only one element but to change many of them.

.. gdot::
    :script: DOT-SECTION
    :process:

    from typing import Any
    import numpy as np
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray

    # The ONNX function
    @onnxnumpy_default
    def onnx_change_element(x: NDArray[Any, np.float32]) -> NDArray[Any, np.float32]:
        shape = x.shape
        v = x.reshape((-1, )).copy()
        v[4] = np.float32(5)
        return v.reshape(shape)

    onx = onnx_change_element.compiled.onnx_
    oinf = onnx_change_element.compiled.rt_fct_.rt
    print("DOT-SECTION", oinf.to_dot())

Instructions using slice is also supported: ``v[:5] = 3.5``, ``v[5:] = 3.5``, ...

Common errors
+++++++++++++

Missing wrapper
^^^^^^^^^^^^^^^

The wrapper intercepts the output of the function and
returns a new function with a runtime. The inner function
returns an instance of type
:class:`OnnxVar <mlprodict.npy.onnx_variable.OnnxVar>`.
It is an layer on the top of ONNX and holds a method doing
the conversion to ONNX :meth:`to_algebra
<mlprodict.npy.onnx_variable.OnnxVar.to_algebra>`.

.. runpython::
    :showcode:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray

    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[Any, np.float32]:
        return npnx.log(x + np.float32(1))

    x = np.random.rand(2, 3).astype(np.float32)
    print(onnx_log_1(x))

The execution does not fail but returns an instance of class
:class:`OnnxVar <mlprodict.npy.onnx_variable.OnnxVar>`. This
instance holds all the necessary information to create the ONNX
graph.

Missing annotation
^^^^^^^^^^^^^^^^^^

The annotation is needed to determine the input and output types.
The runtime would fail executing the ONNX graph without that.

.. runpython::
    :showcode:
    :exception:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray

    @onnxnumpy_default
    def onnx_log_1(x):
        return npnx.log(x + np.float32(1))

Type mismatch
^^^^^^^^^^^^^

As mentioned below, ONNX is strict about types.
If ONNX does an addition, it expects to do it with the same
types. If types are different, one must be cast into the other one.

.. runpython::
    :showcode:
    :exception:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray

    @onnxnumpy_default
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[Any, np.float32]:
        return npnx.log(x + 1)  # -> replace 1 by numpy.float32(1)

    x = np.random.rand(2, 3).astype(np.float32)
    print(onnx_log_1(x))

.. _l-npy-shape-mismatch:

Shape mismatch
^^^^^^^^^^^^^^

The signature of the custom function does not specify any output shape
but the converter of the next transformer in the pipeline might
except one.

.. runpython::
    :showcode:
    :exception:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    @onnxnumpy_default
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[Any, np.float32]:
        return npnx.log(x + np.float32(1))

    data = load_iris()
    X, y = data.data.astype(np.float32), data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe = make_pipeline(
                FunctionTransformer(onnx_log_1),
                StandardScaler(),
                LogisticRegression())
    pipe.fit(X_train, y_train)
    print(pipe.predict_proba(X_test[:2]))

    onx = to_onnx(pipe, X_train[:1], rewrite_ops=True,
                  options={LogisticRegression: {'zipmap': False}})

`NDArray[Any, np.float32]` needs to be replaced by
`NDArray[(None, None), np.float32]` to tell next converter the
output is a two dimension array.

.. _l-npy-missing-converter:

Missing converter
^^^^^^^^^^^^^^^^^

The default converter for *FunctionTransformer* implemented in
:epkg:`sklearn-onnx` does not support custom functions,
only identity, which defeats the purpose of using such preprocessing.
The conversion fails unless the default converter is replaced by
a new one supporting custom functions implemented this API.

.. runpython::
    :showcode:
    :exception:
    :warningout: DeprecationWarning
    :process:

    from typing import Any
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    import mlprodict.npy.numpy_onnx_impl as npnx
    from mlprodict.npy import onnxnumpy_default, NDArray
    from mlprodict.onnx_conv import to_onnx
    from mlprodict.onnxrt import OnnxInference

    @onnxnumpy_default
    def onnx_log_1(x: NDArray[Any, np.float32]) -> NDArray[(None, None), np.float32]:
        return npnx.log(x + np.float32(1))

    data = load_iris()
    X, y = data.data.astype(np.float32), data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe = make_pipeline(
                FunctionTransformer(onnx_log_1),
                StandardScaler(),
                LogisticRegression())
    pipe.fit(X_train, y_train)
    onx = to_onnx(pipe, X_train[:1],
                  options={LogisticRegression: {'zipmap': False}})

There are a couple of ways to fix this example. One way is to call
:func:`to_onnx <mlprodict.onnx_conv.convert.to_onnx>` function with
argument `rewrite_ops=True`. The function restores the default
converter after the call. Another way is to call function
:func:`register_rewritten_operators
<mlprodict.onnx_conv.register_rewritten_converters.register_rewritten_operators>`
but changes are permanent.