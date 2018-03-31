
.. blogpost::
    :title: Similar projects
    :keywords: sklearn-porter, onnx, onnxmltools, winmltools
    :date: 2018-03-31
    :categories: modules

    I would not say this module is actively maintained.
    It was more fun to have the idea, to test it on some
    simple model than to extend its coverage to all available
    models in :epkg:`scikit-learn`. Some altenatives exists
    but it is still ongoing work.
    `sklearn-porter <https://github.com/nok/sklearn-porter>`_
    proposed to produce code into many languages,
    C++, Javascipt, PHP, Java, Ruby, Go. It only includes
    learners and not transforms.
    `onnx <https://github.com/onnx/onnx>`_ proposes to convert
    any models into a unified format. This module implements
    the format,
    `onnxmltools <https://github.com/onnx/onnxmltools>`_,
    `winmltools <https://pypi.python.org/pypi/winmltools>`_
    do the conversion of many models from
    :epkg:`scikit-learn`,
    `xgboost <https://github.com/dmlc/xgboost>`_,
    `lightgbm <https://github.com/Microsoft/LightGBM>`_.
    The produced file can be used to run prediction on GPU
    and Windows with a dedicated runtime.
