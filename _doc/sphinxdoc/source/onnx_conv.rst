
.. _l-onnx-conv:

ONNX New Converters
===================

*mlprodict* implements a couple of converters.
The function :func:`register_converters
<mlprodict.onnx_conv.register.register_converters>` registers
them and makes them visible to :epkg:`sklearn-onnx` so that
a pipeline including one of the supported operators
can be fully converted.

.. runpython::
    :rst:
    :warningout: DeprecationWarning
    :showcode:

    from mlprodict.onnx_conv.register import register_converters
    from pandas import DataFrame
    from pyquickhelper.pandashelper import df2rst
    models = register_converters()
    data = [(cl.__name__,
             dict(name=cl.__name__,
                 module=":epkg:`{}`".format(cl.__module__.split('.')[0])))
            for cl in models]
    data.sort()
    data = [_[1] for _ in data]
    df = DataFrame(data)
    print(df2rst(df))
