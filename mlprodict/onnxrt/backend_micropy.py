"""
@file
@brief ONNX Backend for @see cl OnnxInference.

::

    import unittest
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO
    from onnx.backend.test import BackendTest
    import mlprodict.onnxrt.backend_micropy as backend

    back_test = BackendTest(backend, __name__)
    back_test.exclude('.*_blvc_.*')
    back_test.exclude('.*_densenet_.*')
    back_test.exclude('.*_densenet121_.*')
    back_test.exclude('.*_inception_.*')
    back_test.exclude('.*_resnet50_.*')
    back_test.exclude('.*_shufflenet_.*')
    back_test.exclude('.*_squeezenet_.*')
    back_test.exclude('.*_vgg19_.*')
    back_test.exclude('.*_zfnet512_.*')
    globals().update(back_test.enable_report().test_cases)
    buffer = StringIO()
    print('---------------------------------')

    if True:
        with redirect_stdout(buffer):
            with redirect_stderr(buffer):
                res = unittest.main(verbosity=2, exit=False)
    else:
        res = unittest.main(verbosity=2, exit=False)

    testsRun = res.result.testsRun
    errors = len(res.result.errors)
    skipped = len(res.result.skipped)
    unexpectedSuccesses = len(res.result.unexpectedSuccesses)
    expectedFailures = len(res.result.expectedFailures)
    print('---------------------------------')
    print("testsRun=%d errors=%d skipped=%d unexpectedSuccesses=%d "
          "expectedFailures=%d" % (
        testsRun, errors, skipped, unexpectedSuccesses,
        expectedFailures))
    ran = testsRun - skipped
    print("ratio=%f" % (1 - errors * 1.0 / ran))
    print('---------------------------------')
    lines = buffer.getvalue().split('\\n')
    print("\\n".join(line for line in lines
          if "skipped 'no matched include pattern'" not in line))
"""
from .backend import OnnxInferenceBackendMicro

is_compatible = OnnxInferenceBackendMicro.is_compatible
prepare = OnnxInferenceBackendMicro.prepare
run = OnnxInferenceBackendMicro.run_model
supports_device = OnnxInferenceBackendMicro.supports_device
