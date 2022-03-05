"""
@file
@brief ONNX Backend for @see cl OnnxInference.

::

    import unittest
    from onnx.backend.test import BackendTest
    backend_test = BackendTest(backend, __name__)
    back_test.include('.*add.*')
    globals().update(backend_test.enable_report().test_cases)
    unittest.main()
"""
import unittest
import os
import numpy
from onnx import ModelProto, helper, version
from onnx.checker import check_model
from onnx.backend.base import Backend, BackendRep
from .onnx_inference import OnnxInference


class OnnxInferenceBackendRep(BackendRep):
    """
    Computes the prediction for an ONNX graph
    loaded with @see cl OnnxInference.

    :param session: @see cl OnnxInference
    """

    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        """
        Computes the prediction. See @see meth OnnxInference.run.
        """
        if isinstance(inputs, list):
            feeds = {}
            for i, inp in enumerate(self._session.input_names):
                feeds[inp] = inputs[i]
        elif isinstance(inputs, dict):
            feeds = inputs
        elif isinstance(inputs, numpy.ndarray):
            names = self._session.input_names
            if len(names) != 1:
                raise RuntimeError(
                    "Expecting one input not %d." % len(names))
            feeds = {names[0]: inputs}
        else:
            raise TypeError(
                "Unexpected input type %r." % type(inputs))
        outs = self._session.run(feeds)
        return [outs[name] for name in self._session.output_names]


class OnnxInferenceBackend(Backend):
    """
    ONNX backend following the pattern from
    `onnx/backend/base.py
    <https://github.com/onnx/onnx/blob/main/onnx/backend/base.py>`_.
    """

    @classmethod
    def is_compatible(cls, model, device=None, **kwargs):
        """
        Returns whether the model is compatible with the backend.

        :param model: unused
        :param device: None to use the default device or a string (ex: `'CPU'`)
        :return: boolean
        """
        return device is None or device == 'CPU'

    @classmethod
    def is_opset_supported(cls, model):
        """
        Returns whether the opset for the model is supported by the backend.

        :param model: Model whose opsets needed to be verified.
        :return: boolean and error message if opset is not supported.
        """
        return True, ''

    @classmethod
    def supports_device(cls, device):
        """
        Checks whether the backend is compiled with particular
        device support.
        """
        return device == 'CPU'

    @classmethod
    def create_inference_session(cls, model):
        return OnnxInference(model)

    @classmethod
    def prepare(cls, model, device=None, **kwargs):
        """
        Loads the model and creates @see cl OnnxInference.

        :param model: ModelProto (returned by `onnx.load`),
            string for a filename or bytes for a serialized model
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see @see cl OnnxInference
        :return: see @see cl OnnxInference
        """
        if isinstance(model, OnnxInferenceBackendRep):
            return model
        if isinstance(model, OnnxInference):
            return OnnxInferenceBackendRep(model)
        if isinstance(model, (str, bytes)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)

        onnx_version = tuple(map(int, (version.version.split(".")[:3])))
        onnx_supports_serialized_model_check = onnx_version >= (1, 10, 0)
        bin_or_model = (
            model.SerializeToString() if onnx_supports_serialized_model_check
            else model)
        check_model(bin_or_model)
        opset_supported, error_message = cls.is_opset_supported(model)
        if not opset_supported:
            raise unittest.SkipTest(error_message)
        bin = bin_or_model
        if not isinstance(bin, (str, bytes)):
            bin = bin.SerializeToString()
        return cls.prepare(bin, device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        """
        Computes the prediction.

        :param model: see @see cl OnnxInference returned by function *prepare*
        :param inputs: inputs
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see @see cl OnnxInference
        :return: predictions
        """
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        '''
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        '''
        raise NotImplementedError("Unable to run the model node by node.")


class OnnxInferenceBackendOrt(OnnxInferenceBackend):
    
    @classmethod
    def create_inference_session(cls, model):
        return OnnxInference(model, runtime='onnxruntime1')
