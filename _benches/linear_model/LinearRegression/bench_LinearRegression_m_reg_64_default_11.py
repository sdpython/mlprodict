import numpy  # pylint: disable=W0611
# Import specific to this model.
from sklearn.linear_model import LinearRegression

from mlprodict.asv_benchmark import _CommonAsvSklBenchmarkRegressor
from mlprodict.onnx_conv import to_onnx  # pylint: disable=W0611
from mlprodict.onnxrt import OnnxInference  # pylint: disable=W0611


class LinearRegression_m_reg_64_default_11_benchRegressor(_CommonAsvSklBenchmarkRegressor):
    """
    :epkg:`asv` example for a regressor,
    Full template can be found in
    `common_asv_skl.py <https://github.com/sdpython/mlprodict/
    blob/master/mlprodict/asv_benchmark/common_asv_skl.py>`_.
    """

    params = [
        ['skl', 'pyrt'],
        (1, 100, 10000),
        (4, 20),
    ]
    param_names = ['rt', 'N', 'nf']
    xtest_dtype = numpy.float64
    target_opset = 11

    def _create_model(self):
        return LinearRegression()
