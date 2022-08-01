
ONNX Backends for Python/Numpy runtime (compiled)
=================================================

Backend class: :class:`OnnxInferenceBackend
<mlprodict.onnxrt.backend.OnnxInferenceBackend>`.

.. runpython::
    :showcode:
    :process:

    import unittest
    import sys
    from datetime import datetime
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO
    from onnx.backend.test import BackendTest
    from onnx import __version__ as onnx_version
    from onnxruntime import __version__ as ort_version
    from numpy import __version__ as npy_version
    import mlprodict.onnxrt.backend_pyc as backend

    back_test = BackendTest(backend, __name__)
    back_test.include('.*_cpu')
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

    print('---------------------------------')
    print('python', sys.version)
    print('onnx', onnx_version)
    print('onnxruntime', ort_version)
    print('numpy', npy_version)
    print('---------------------------------')
    print(datetime.now(), "BEGIN")
    print('---------------------------------')

    buffer = StringIO()
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
    print(datetime.now(), "END")
    print('---------------------------------')

    print("testsRun=%d errors=%d skipped=%d" % (testsRun, errors, skipped))
    print("unexpectedSuccesses=%d expectedFailures=%d" % (
        unexpectedSuccesses, expectedFailures))
    ran = testsRun - skipped
    print("ratio=%f" % (1 - errors * 1.0 / ran))
    print('---------------------------------')
    lines = buffer.getvalue().split('\n')
    print("\n".join(line for line in lines
          if "skipped 'no matched include pattern'" not in line))
