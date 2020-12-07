"""
@file
@brief Measures time processing for ONNX models.
"""
import pickle
import os
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail  # pylint: disable=E0611
import sklearn
from .. import OnnxInference
from .validate_helper import default_time_kwargs, measure_time, _multiply_time_kwargs
from .validate_benchmark import make_n_rows


class SimplifiedOnnxInference:
    "Simple wrapper around InferenceSession which imitates OnnxInference."

    def __init__(self, ort):
        from onnxruntime import InferenceSession
        self.sess = InferenceSession(ort)

    @property
    def input_names(self):
        "Returns InferenceSession input names."
        return [_.name for _ in self.sess.get_inputs()]

    def run(self, input):
        "Calls InferenceSession.run."
        return self.sess.run(None, input)


def enumerate_benchmark_replay(folder, runtime='python', time_kwargs=None,
                               skip_long_test=True, time_kwargs_fact=None,
                               time_limit=4, verbose=1, fLOG=None):
    """
    Replays a benchmark stored with function
    :func:`enumerate_validated_operator_opsets
    <mlprodict.onnxrt.validate.validate.enumerate_validated_operator_opsets>`
    or command line :ref:`validate_runtime <l-cmd-validate_runtime>`.
    Enumerates the results.

    @param      folder              folder where to find pickled files, all files must have
                                    *pkl* or *pickle* extension
    @param      runtime             runtime or runtimes
    @param      time_kwargs         to define a more precise way to measure a model
    @param      skip_long_test      skips tests for high values of N if they seem too long
    @param      time_kwargs_fact    see :func:`_multiply_time_kwargs <mlprodict.onnxrt.validate.validate_helper._multiply_time_kwargs>`
    @param      time_limit          to skip the rest of the test after this limit (in second)
    @param      verbose             if >= 1, uses :epkg:`tqdm`
    @param      fLOG                logging function
    @return                         iterator on results
    """
    files = [_ for _ in os.listdir(folder) if _.endswith(
        ".pkl") or _.endswith("_.pickle")]
    if len(files) == 0:
        raise FileNotFoundError(
            "Unable to find any file in folder '{}'.".format(folder))

    if time_kwargs in (None, ''):
        time_kwargs = default_time_kwargs()

    if isinstance(runtime, str):
        runtime = runtime.split(",")

    loop = files
    if verbose >= 1:
        try:
            from tqdm import tqdm
            loop = tqdm(files)
        except ImportError:  # pragma: no cover
            pass

    for pkl in loop:
        if "ERROR" in pkl:
            # An error.
            if verbose >= 2 and fLOG is not None:  # pragma: no cover
                fLOG(  # pragma: no cover
                    "[enumerate_benchmark_replay] skip '{}'.".format(pkl))
            continue  # pragma: no cover
        if verbose >= 2 and fLOG is not None:
            fLOG("[enumerate_benchmark_replay] process '{}'.".format(pkl))
        row = {}
        with open(os.path.join(folder, pkl), 'rb') as f:
            obj = pickle.load(f)
        X_test = obj['X_test']
        ort_test = obj['Xort_test']
        onx = obj['onnx_bytes']
        model = obj['skl_model']
        tkw = _multiply_time_kwargs(time_kwargs, time_kwargs_fact, model)
        row['folder'] = folder
        row['filename'] = pkl
        row['n_features'] = X_test.shape[1]

        for key in ['assume_finite', 'conv_options',
                    'init_types', 'idtype', 'method_name', 'n_features',
                    'name', 'optim', 'opset', 'predict_kwargs',
                    'output_index', 'problem', 'scenario']:
            row[key] = obj['obs_op'][key]

        # 'bench-batch',
        # 'bench-skl',

        oinfs = {}
        for rt in runtime:
            if rt == 'onnxruntime':
                try:
                    oinfs[rt] = SimplifiedOnnxInference(onx)
                except (OrtFail, RuntimeError) as e:  # pragma: no cover
                    row['ERROR'] = str(e)
                    oinfs[rt] = None
            else:
                try:
                    oinfs[rt] = OnnxInference(onx, runtime=rt)
                except (OrtFail, RuntimeError) as e:  # pragma: no cover
                    row['ERROR'] = str(e)
                    oinfs[rt] = None

        for k, v in sorted(tkw.items()):
            if verbose >= 3 and fLOG is not None:
                fLOG(  # pragma: no cover
                    "[enumerate_benchmark_replay] process n_rows={} - {}".format(k, v))
            xt = make_n_rows(X_test, k)
            number = v['number']
            repeat = v['repeat']

            meth = getattr(model, row['method_name'])
            with sklearn.config_context(assume_finite=row['assume_finite']):
                skl = measure_time(lambda x: meth(x), xt,
                                   number=number, repeat=repeat,
                                   div_by_number=True)
            if verbose >= 4 and fLOG is not None:
                fLOG(  # pragma: no cover
                    "[enumerate_benchmark_replay] skl={}".format(skl))
            row['%d-skl-details' % k] = skl
            row['%d-skl' % k] = skl['average']

            xto = make_n_rows(ort_test, k)
            for rt in runtime:
                oinf = oinfs[rt]
                if oinf is None:
                    continue  # pragma: no cover
                if len(oinf.input_names) != 1:
                    raise NotImplementedError(  # pragma: no cover
                        "This function only allows one input not {}".format(
                            len(oinf.input_names)))
                name = oinf.input_names[0]
                ort = measure_time(lambda x: oinf.run({name: x}), xto,
                                   number=number, repeat=repeat,
                                   div_by_number=True)
                if verbose >= 4 and fLOG is not None:
                    fLOG(  # pragma: no cover
                        "[enumerate_benchmark_replay] {}={}".format(rt, ort))
                row['%d-%s-detail' % (k, rt)] = ort
                row['%d-%s' % (k, rt)] = ort['average']
        yield row
