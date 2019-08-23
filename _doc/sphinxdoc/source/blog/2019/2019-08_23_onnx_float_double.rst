
.. blogpost::
    :title: Float, double with ONNX
    :keywords: onnx, float, double
    :date: 2019-08-23
    :categories: onnx

    Replicating what a library does, :epkg:`scikit-learn` for
    example, is different from implementing a function
    defined in a paper. Every trick needs to be replicated.
    :epkg:`scikit-learn` trees implement a prediction function
    which takes float features and compares them to double
    thresholds. Knowning the :epkg:`ONNX` assumes that comparison
    only happens numbers of the same type, you end up with discrepencies.

    To be honest, I did not imagine that :epkg:`scikit-learn` would
    mix types when comparing numbers, I just assumed it was using
    double as many other predictors do. As none of my other ideas
    seemed to work, I went back to :epkg:`scikit-learn` code,
    and discovered this mixed type comparison. It makes sense
    as it uses less memory and probably has a small impact on
    the performance. From an ONNX point of view, I had to change
    the thresholds to replicate :epkg:`scikit-learn` behaviour.
    Notebook :ref:`onnxfloatdoubleskldecisiontreesrst` shows how
    I handled it.
