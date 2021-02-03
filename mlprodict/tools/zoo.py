"""
@file
@brief Tools to test models from the :epkg:`ONNX Zoo`.

.. versionadded:: 0.6
"""
import os
import urllib.request
from collections import OrderedDict
import numpy
from onnx import TensorProto, numpy_helper


def short_list_zoo_models():
    """
    Returns a short list from :epkg:`ONNX Zoo`.

    :return: list of dictionaries.

    .. runpython::
        :showcode:

        import pprint
        from mlprodict.tools.zoo import short_list_zoo_models
        pprint.pprint(short_list_zoo_models())
    """
    return [
        dict(name="mobilenet",
             model="https://github.com/onnx/models/raw/master/vision/"
                   "classification/mobilenet/model/mobilenetv2-7.tar.gz"),
        dict(name="resnet18",
             model="https://github.com/onnx/models/raw/master/vision/"
                   "classification/resnet/model/resnet18-v1-7.tar.gz"),
        dict(name="squeezenet",
             model="https://github.com/onnx/models/raw/master/vision/"
                   "classification/squeezenet/model/squeezenet1.0-9.tar.gz"),
        dict(name="densenet121",
             model="https://github.com/onnx/models/raw/master/vision/"
                   "classification/densenet-121/model/densenet-9.tar.gz"),
        dict(name="inception2",
             model="https://github.com/onnx/models/raw/master/vision/"
                   "classification/inception_and_googlenet/inception_v2/"
                   "model/inception-v2-9.tar.gz"),
        dict(name="shufflenet",
             model="https://github.com/onnx/models/raw/master/vision/"
                   "classification/shufflenet/model/shufflenet-9.tar.gz"),
        dict(name="efficientnet-lite4",
             model="https://github.com/onnx/models/raw/master/vision/"
                   "classification/efficientnet-lite4/model/"
                   "efficientnet-lite4-11.tar.gz"),
    ]


def _download_url(url, output_path, name, verbose=False):
    if verbose:
        from tqdm import tqdm

        class DownloadProgressBar(tqdm):
            "progress bar hook"

            def update_to(self, b=1, bsize=1, tsize=None):
                "progress bar hook"
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=name) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to)
    else:
        urllib.request.urlretrieve(url, filename=output_path)


def load_data(folder):
    """
    Restores protobuf data stored in a folder.

    :param folder: folder
    :return: dictionary
    """
    res = OrderedDict()
    files = os.listdir(folder)
    for name in files:
        noext, ext = os.path.splitext(name)
        if ext == '.pb':
            data = TensorProto()
            with open(os.path.join(folder, name), 'rb') as f:
                data.ParseFromString(f.read())
            res[noext] = numpy_helper.to_array(data)

    return res


def download_model_data(name, model=None, cache=None, verbose=False):
    """
    Downloads a model and returns a link to the local
    :epkg:`ONNX` file and data which can be used as inputs.

    :param name: model name (see @see fn short_list_zoo_models)
    :param model: url or empty to get the default value
        returned by @see fn short_list_zoo_models)
    :param cache: folder to cache the downloaded data
    :param verbose: display a progress bar
    :return: local onnx file, input data
    """
    if model is None:
        model_list = short_list_zoo_models()
        for mod in model_list:
            if mod['name'] == name:
                model = mod['model']
                break
        if model is None:
            raise ValueError(
                "Unable to find a default value for name=%r." % name)

    # downloads
    last_name = model.split('/')[-1]
    if cache is None:
        cache = os.path.abspath('.')
    dest = os.path.join(cache, last_name)
    if not os.path.exists(dest):
        _download_url(model, dest, name, verbose=verbose)
    size = os.stat(dest).st_size
    if size < 2 ** 20:  # pragma: no cover
        os.remove(dest)
        raise RuntimeError(
            "Unable to download model from %r." % model)

    outtar = os.path.splitext(dest)[0]
    if not os.path.exists(outtar):
        from pyquickhelper.filehelper.compression_helper import (
            ungzip_files)
        ungzip_files(dest, unzip=False, where_to=cache, remove_space=False)

    onnx_file = os.path.splitext(outtar)[0]
    if not os.path.exists(onnx_file):
        from pyquickhelper.filehelper.compression_helper import (
            untar_files)
        untar_files(outtar, where_to=cache)

    onnx_files = [_ for _ in os.listdir(onnx_file) if _.endswith(".onnx")]
    if len(onnx_files) != 1:
        raise FileNotFoundError(  # pragma: no cover
            "Unable to find any onnx file in %r." % onnx_files)
    final_onnx = os.path.join(onnx_file, onnx_files[0])

    # data
    data = [_ for _ in os.listdir(onnx_file)
            if os.path.isdir(os.path.join(onnx_file, _))]
    examples = OrderedDict()
    for f in data:
        examples[f] = load_data(os.path.join(onnx_file, f))

    return final_onnx, examples


def verify_model(onnx_file, examples, runtime=None, abs_tol=5e-4):
    """
    Verifies a model.

    :param onnx_file: ONNX file
    :param examples: list of examples to verify
    :param runtime: a runtime to use
    :param abs_tol: error tolerance when checking the output
    :return: errors for every sample
    """
    if runtime == 'onnxruntime':
        from onnxruntime import InferenceSession
        sess = InferenceSession(onnx_file)
        meth = lambda data, s=sess: s.run(None, data)
        names = [p.name for p in sess.get_inputs()]
        onames = list(range(len(sess.get_outputs())))
    else:
        def _lin_(sess, data, names):
            r = sess.run(data)
            return [r[n] for n in names]

        from ..onnxrt import OnnxInference
        sess = OnnxInference(onnx_file, runtime=runtime)
        names = sess.input_names
        onames = sess.output_names
        meth = lambda data, s=sess, ns=onames: _lin_(s, data, ns)

    rows = []
    for index, (name, data) in enumerate(examples.items()):
        inputs = {n: data[v] for n, v in zip(names, data)}
        outputs = meth(inputs)
        data_values = list(data.items())
        expected = [d[1] for d in data_values[len(inputs):]]
        if len(outputs) != len(onames):
            raise RuntimeError(
                "Number of outputs %d is != expected outputs %d." % (
                    len(outputs), len(onames)))
        for i in range(len(outputs)):  # pylint: disable=C0200
            if outputs[i].shape != expected[i].shape:
                raise ValueError(
                    "Shape mismatch got %r != expected %r." % (
                        outputs[i].shape, expected[i].shape))
            diff = numpy.abs(outputs[i] - expected[i]).ravel()
            absolute = diff.max()
            relative = absolute / numpy.median(diff) if absolute > 0 else 0.
            if absolute > abs_tol:
                raise ValueError(
                    "Example %d, inferred and expected resuls are different "
                    "for output %d: abs=%r rel=%r (runtime=%r)."
                    "" % (index, i, absolute, relative, runtime))
            rows.append(dict(name=name, i=i, abs=absolute, rel=relative))
    return rows
