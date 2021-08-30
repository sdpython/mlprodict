
scikit-learn API and ONNX graph in pipelines
============================================

This is the main class which makes it easy to insert
to use the prediction from an :epkg:`ONNX` files into a :epkg:`scikit-learn`
pipeline.

.. contents::
    :local:

OnnxPipeline
++++++++++++

.. autosignature:: mlprodict.sklapi.onnx_pipeline.OnnxPipeline
    :members:

OnnxTransformer
+++++++++++++++

.. autosignature:: mlprodict.sklapi.onnx_transformer.OnnxTransformer
    :members:

SpeedUp scikit-learn pipeline with ONNX
+++++++++++++++++++++++++++++++++++++++

These classes wraps an existing pipeline from *scikit-learn*
and replaces the inference (*transform*, *predict*, *predict_proba*)
by another runtime built after the model was converted into ONNX.
See example :ref:`l-b-numpy-numba-ort` for further details.

.. autosignature:: mlprodict.sklapi.onnx_speed_up.OnnxSpeedUpClassifier
    :members:

.. autosignature:: mlprodict.sklapi.onnx_speed_up.OnnxSpeedUpRegressor
    :members:

.. autosignature:: mlprodict.sklapi.onnx_speed_up.OnnxSpeedUpTransformer
    :members:
