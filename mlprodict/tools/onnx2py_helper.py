"""
@file
@brief Functions which converts :epkg:`ONNX` object into
readable :epkg:`python` objects.
"""
import pprint
import warnings
import numpy
from scipy.sparse import coo_matrix
from onnx import onnx_pb as onnx_proto, TensorProto
from onnx.numpy_helper import to_array, from_array


def to_bytes(val):
    """
    Converts an array into protobuf and then into bytes.

    @param      val     array
    @return             bytes

    .. exref::
        :title: Converts an array into bytes (serialization)

        Useful to serialize.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning

            import numpy
            from mlprodict.tools.onnx2py_helper import to_bytes

            data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.float32)
            pb = to_bytes(data)
            print(len(pb), data.size * data.itemsize, pb[:10])
    """
    if isinstance(val, numpy.ndarray):
        pb = from_array(val)
    else:
        pb = val  # pragma: no cover
    return pb.SerializeToString()


def from_bytes(b):
    """
    Retrieves an array from bytes then protobuf.

    @param      b       bytes
    @return             array

    .. exref::
        :title: Converts bytes into an array (serialization)

        Useful to deserialize.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning

            import numpy
            from mlprodict.tools.onnx2py_helper import to_bytes, from_bytes

            data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.float32)
            pb = to_bytes(data)
            data2 = from_bytes(pb)
            print(data2)
    """
    if isinstance(b, bytes):
        pb = TensorProto()
        pb.ParseFromString(b)
    else:
        pb = b  # pragma: no cover
    return to_array(pb)


def _numpy_array(data, dtype=None, copy=True):
    """
    Single function to create an array.

    @param      data        data
    @param      dtype       dtype
    @param      copy        copy
    @return                 numpy array
    """
    if isinstance(data, numpy.ndarray):
        res = data
    else:
        res = numpy.array(data, dtype=dtype, copy=copy)
    return res


def _sparse_array(shape, data, indices, dtype=None, copy=True):
    """
    Single function to create an sparse array
    (:epkg:`coo_matrix`).

    @param      shape       shape
    @param      data        data
    @param      indices     indices
    @param      dtype       dtype
    @param      copy        copy
    @return                 :epkg:`coo_matrix`
    """
    if len(shape) != 2:
        raise ValueError(  # pragma: no cover
            "Only matrices are allowed or sparse matrices "
            "but shape is {}.".format(shape))
    rows = numpy.array([i // shape[1] for i in indices])
    cols = numpy.array([i % shape[1] for i in indices])
    if isinstance(data, numpy.ndarray):
        res = coo_matrix((data, (rows, cols)), dtype=dtype)
    else:
        res = coo_matrix(  # pragma: no cover
            (numpy.array(data, dtype=dtype, copy=copy),
             (rows, cols)), dtype=dtype)
    return res


def _elem_type_as_str(elem_type):
    if elem_type == onnx_proto.TensorProto.FLOAT:  # pylint: disable=E1101
        return 'float'
    if elem_type == onnx_proto.TensorProto.BOOL:  # pylint: disable=E1101
        return 'bool'
    if elem_type == onnx_proto.TensorProto.DOUBLE:  # pylint: disable=E1101
        return 'double'
    if elem_type == onnx_proto.TensorProto.STRING:  # pylint: disable=E1101
        return 'str'
    if elem_type == onnx_proto.TensorProto.INT64:  # pylint: disable=E1101
        return 'int64'
    if elem_type == onnx_proto.TensorProto.INT32:  # pylint: disable=E1101
        return 'int32'
    if elem_type == onnx_proto.TensorProto.UINT32:  # pylint: disable=E1101
        return 'uint32'
    if elem_type == onnx_proto.TensorProto.UINT64:  # pylint: disable=E1101
        return 'uint64'
    if elem_type == onnx_proto.TensorProto.INT16:  # pylint: disable=E1101
        return 'int16'
    if elem_type == onnx_proto.TensorProto.UINT16:  # pylint: disable=E1101
        return 'uint16'
    if elem_type == onnx_proto.TensorProto.UINT8:  # pylint: disable=E1101
        return 'uint8'
    if elem_type == onnx_proto.TensorProto.INT8:  # pylint: disable=E1101
        return 'int8'
    if elem_type == onnx_proto.TensorProto.FLOAT16:  # pylint: disable=E1101
        return 'float16'
    if elem_type == 0:  # pylint: disable=E1101
        return 'unk'

    # The following code should be refactored.
    selem = str(elem_type)

    if selem.startswith("tensor_type"):
        this = elem_type.tensor_type
        et = _elem_type_as_str(this.elem_type)
        shape = this.shape
        dim = shape.dim
        dims = [d.dim_value for d in dim]
        if len(dims) == 0:
            dims = '?'
        return {'kind': 'tensor', 'elem': et, 'shape': shape}

    if selem.startswith("map_type"):
        this = elem_type.map_type
        kt = _elem_type_as_str(this.key_type)
        vt = _elem_type_as_str(this.value_type)
        return {'kind': 'map', 'key': kt, 'value': vt}

    raise NotImplementedError(  # pragma: no cover
        "elem_type '{}' is unknown\nfields:\n{}\n-----\n{}.".format(
            elem_type, pprint.pformat(dir(elem_type)), type(elem_type)))


def _to_array(var):
    try:
        data = to_array(var)
    except ValueError as e:  # pragma: no cover
        dims = [d for d in var.dims]
        if var.data_type == 1 and var.float_data is not None:
            try:
                data = _numpy_array(var.float_data, dtype=numpy.float32,
                                    copy=False).reshape(dims)
            except ValueError:
                data = _numpy_array(to_array(var))
        elif var.data_type == 11 and var.double_data is not None:
            try:
                data = _numpy_array(var.double_data, dtype=numpy.float64,
                                    copy=False).reshape(dims)
            except ValueError:
                data = _numpy_array(to_array(var))
        elif var.data_type == 6 and var.int32_data is not None:
            data = _numpy_array(var.int32_data, dtype=numpy.int32,
                                copy=False).reshape(dims)
        elif var.data_type == 7 and var.int64_data is not None:
            data = _numpy_array(var.int64_data, dtype=numpy.int64,
                                copy=False).reshape(dims)
        elif var.data_type == 10 and var.float16_data is not None:
            data = _numpy_array(var.float16_data, dtype=numpy.float16,
                                copy=False).reshape(dims)
        else:
            raise NotImplementedError(
                "Iniatilizer {} cannot be converted into a dictionary.".format(var)) from e
    return data


def _var_as_dict(var):
    """
    Converts a protobuf object into something readable.
    The current implementation relies on :epkg:`json`.
    That's not the most efficient way.
    """
    if hasattr(var, 'type') and str(var.type) != '':
        # variable
        if var.type is not None:
            if hasattr(var, 'sparse_tensor') and var.type == 11:
                # sparse tensor
                t = var.sparse_tensor
                values = _var_as_dict(t.values)
                dims = list(t.dims)
                dtype = dict(kind='sparse_tensor', shape=tuple(dims), elem=1)
            elif hasattr(var.type, 'tensor_type') and var.type.tensor_type.elem_type > 0:
                t = var.type.tensor_type
                elem_type = _elem_type_as_str(t.elem_type)
                shape = t.shape
                dim = shape.dim
                dims = [d.dim_value for d in dim]
                if len(dims) == 0:
                    dims = '?'
                dtype = dict(kind='tensor', elem=elem_type,
                             shape=tuple(dims))
            elif hasattr(var.type, 'real') and var.type.real == 5 and hasattr(var, 'g'):
                dtype = dict(kind='graph', elem=var.type.real)
            elif hasattr(var.type, 'real') and var.type.real == 4 and hasattr(var, 't'):
                dtype = dict(kind='tensor', elem=var.type.real)
            elif hasattr(var.type, 'real'):
                dtype = dict(kind='real', elem=var.type.real)
            elif (hasattr(var.type, "sequence_type") and var.type.sequence_type is not None and
                    str(var.type.sequence_type.elem_type) != ''):
                t = var.type.sequence_type
                elem_type = _elem_type_as_str(t.elem_type)
                dtype = dict(kind='sequence', elem=elem_type)
            elif (hasattr(var.type, "map_type") and var.type.map_type is not None and
                    str(var.type.map_type.key_type) != '' and
                    str(var.type.map_type.value_type) != ''):
                t = var.type.map_type
                key_type = _elem_type_as_str(t.key_type)
                value_type = _elem_type_as_str(t.value_type)
                dtype = dict(kind='map', key=key_type, value=value_type)
            elif hasattr(var.type, 'tensor_type') and var.type.tensor_type.elem_type == 0:
                t = var.type.tensor_type
                elem_type = _elem_type_as_str(t.elem_type)
                shape = t.shape
                dim = shape.dim
                dims = [d.dim_value for d in dim]
                if len(dims) == 0:
                    dims = '?'
                dtype = dict(kind='tensor', elem=elem_type,
                             shape=tuple(dims))
            else:
                raise NotImplementedError(  # pragma: no cover
                    "Unable to convert a type into a dictionary for '{}'. "
                    "Available fields: {}.".format(
                        var.type, pprint.pformat(dir(var.type))))
        else:
            raise NotImplementedError(  # pragma: no cover
                "Unable to convert variable into a dictionary for '{}'. "
                "Available fields: {}.".format(
                    var, pprint.pformat(dir(var.type))))

        res = dict(name=var.name, type=dtype)

        if (hasattr(var, 'sparse_tensor') and dtype.get('elem', None) == 1 and
                dtype['kind'] == 'sparse_tensor'):
            # sparse matrix
            t = var.sparse_tensor
            try:
                values = _var_as_dict(t.values)
            except NotImplementedError as e:  # pragma: no cover
                raise NotImplementedError(
                    "Issue with\n{}\n---".format(var)) from e
            indices = _var_as_dict(t.indices)
            res['value'] = _sparse_array(
                dtype['shape'], values['value'], indices['value'], dtype=numpy.float32)
        elif hasattr(var, 'floats') and dtype.get('elem', None) == 6:
            res['value'] = _numpy_array(var.floats, dtype=numpy.float32)
        elif hasattr(var, 'strings') and dtype.get('elem', None) == 8:
            res['value'] = _numpy_array(var.strings)
        elif hasattr(var, 'ints') and dtype.get('elem', None) == 7:
            res['value'] = _numpy_array(var.ints)
        elif hasattr(var, 'f') and dtype.get('elem', None) == 1:
            res['value'] = var.f
        elif hasattr(var, 's') and dtype.get('elem', None) == 3:
            res['value'] = var.s
        elif hasattr(var, 'i') and dtype.get('elem', None) == 2:
            res['value'] = var.i
        elif hasattr(var, 'g') and dtype.get('elem', None) == 5:
            res['value'] = var.g
        elif hasattr(var, 't') and dtype.get('elem', None) == 4:
            ts = _var_as_dict(var.t)
            res['value'] = ts['value']
        elif hasattr(var, 'sparse_tensor') and dtype.get('elem', None) == 11:
            ts = _var_as_dict(var.sparse_tensor)
            res['value'] = ts['value']
        elif "'value'" in str(var):
            warnings.warn("No value: {} -- {}".format(  # pragma: no cover
                dtype, str(var).replace("\n", "").replace(" ", "")))
        return res

    if hasattr(var, 'op_type'):
        if hasattr(var, 'attribute'):
            atts = {}
            for att in var.attribute:
                atts[att.name] = _var_as_dict(att)
        return dict(name=var.name, op_type=var.op_type,
                    domain=var.domain, atts=atts)
    if hasattr(var, 'dims') and len(var.dims) > 0:
        # initializer
        data = _to_array(var)
        return dict(name=var.name, value=data)
    if hasattr(var, 'data_type') and var.data_type > 0:
        data = _to_array(var)
        return dict(name=var.name, value=data)
    raise NotImplementedError(  # pragma: no cover
        "Unable to guess which object it is.\n{}\n---".format(var))


def _type_to_string(dtype):
    """
    Converts a type into a readable string.
    """
    if not isinstance(dtype, dict):
        dtype_ = _var_as_dict(dtype)  # pragma: no cover
    else:
        dtype_ = dtype
    if dtype_["kind"] == 'tensor':
        return "{0}({1})".format(dtype_['elem'], dtype_['shape'])
    if dtype_['kind'] == 'sequence':
        return "[{0}]".format(_type_to_string(dtype_['elem']))
    if dtype_["kind"] == 'map':
        return "{{{0}, {1}}}".format(dtype_['key'], dtype_['value'])
    raise NotImplementedError(  # pragma: no cover
        "Unable to convert into string {} or {}.".format(dtype, dtype_))


def numpy_min(x):
    """
    Returns the minimum of an array.
    Deals with text as well.
    """
    try:
        if hasattr(x, 'todense'):
            x = x.todense()
        if x.dtype.kind not in 'cUC':
            return x.min()
        try:  # pragma: no cover
            x = x.ravel()
        except AttributeError:  # pragma: no cover
            pass
        keep = list(filter(lambda s: isinstance(s, str), x))
        if len(keep) == 0:  # pragma: no cover
            return numpy.nan
        keep.sort()
        val = keep[0]
        if len(val) > 10:  # pragma: no cover
            val = val[:10] + '...'
        return "%r" % val
    except (ValueError, TypeError):  # pragma: no cover
        return '?'


def numpy_max(x):
    """
    Returns the maximum of an array.
    Deals with text as well.
    """
    try:
        if hasattr(x, 'todense'):
            x = x.todense()
        if x.dtype.kind not in 'cUC':
            return x.max()
        try:  # pragma: no cover
            x = x.ravel()
        except AttributeError:  # pragma: no cover
            pass
        keep = list(filter(lambda s: isinstance(s, str), x))
        if len(keep) == 0:  # pragma: no cover
            return numpy.nan
        keep.sort()
        val = keep[-1]
        if len(val) > 10:  # pragma: no cover
            val = val[:10] + '...'
        return "%r" % val
    except (ValueError, TypeError):  # pragma: no cover
        return '?'


def guess_proto_dtype(dtype):
    """
    Guesses the ONNX dtype given a numpy dtype.

    :param dtype: numpy dtype
    :return: proto type
    """
    if dtype == numpy.float32:
        return TensorProto.FLOAT  # pylint: disable=E1101
    if dtype == numpy.float64:
        return TensorProto.DOUBLE  # pylint: disable=E1101
    if dtype == numpy.int64:
        return TensorProto.INT64  # pylint: disable=E1101
    if dtype == numpy.int32:
        return TensorProto.INT32  # pylint: disable=E1101
    if dtype == numpy.int16:
        return TensorProto.INT16  # pylint: disable=E1101
    if dtype == numpy.int8:
        return TensorProto.INT8  # pylint: disable=E1101
    if dtype == numpy.uint64:
        return TensorProto.UINT64  # pylint: disable=E1101
    if dtype == numpy.uint32:
        return TensorProto.UINT32  # pylint: disable=E1101
    if dtype == numpy.uint16:
        return TensorProto.UINT16  # pylint: disable=E1101
    if dtype == numpy.uint8:
        return TensorProto.UINT8  # pylint: disable=E1101
    if dtype == numpy.float16:
        return TensorProto.FLOAT16  # pylint: disable=E1101
    if dtype in (numpy.bool, bool, numpy.bool_):
        return TensorProto.BOOL  # pylint: disable=E1101
    if dtype in (numpy.str, str, numpy.str_):
        return TensorProto.STRING  # pylint: disable=E1101
    raise RuntimeError(
        "Unable to guess type for dtype={}.".format(dtype))  # pragma: no cover
