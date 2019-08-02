
.. blogpost::
    :title: ONNX updates
    :keywords: onnx, onnxrt, update
    :date: 2019-08-02
    :categories: onnx

    The python runtime is now almost complete for
    all the supported numerical operator implemented
    in :epkg:`sklearn-onnx`. A couple of notebooks
    introduces a couple of way to investigates issues,
    to benchmark ONNX models with :epkg:`onnxruntime`
    or python runtime, to check the differences between
    the same model. It also extend ONNX with operators not in
    the specification to experiment some assumptions
    and check it is more efficient. Notebook
    :ref:`onnxshakerrst` introduces a way to guess the
    margins introduced by the conversion from double
    to single. There also exists a function to convert numpy
    function into ONNX (see :ref:`l-numpy2onnx-tutorial`).
    Its coverage is probably low but it will improve.
