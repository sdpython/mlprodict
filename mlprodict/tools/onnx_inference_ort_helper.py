# pylint: disable=C0302
"""
@file
@brief Helpers for :epkg:`onnxruntime`.
"""
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611,W0611
    OrtDevice as C_OrtDevice)


def get_ort_device(device):
    """
    Converts device into :epkg:`C_OrtDevice`.

    :param device: any type
    :return: :epkg:`C_OrtDevice`

    Example:

    ::

        get_ort_device('cpu')
        get_ort_device('gpu')
        get_ort_device('cuda')
        get_ort_device('cuda:0')
    """
    if isinstance(device, C_OrtDevice):
        return device
    if isinstance(device, str):
        if device == 'cpu':
            return C_OrtDevice(
                C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
        if device in {'gpu', 'cuda:0', 'cuda', 'gpu:0'}:
            return C_OrtDevice(
                C_OrtDevice.cuda(), C_OrtDevice.default_memory(), 0)
        if device.startswith('gpu:'):
            idx = int(device[4:])
            return C_OrtDevice(
                C_OrtDevice.cuda(), C_OrtDevice.default_memory(), idx)
        if device.startswith('cuda:'):
            idx = int(device[5:])
            return C_OrtDevice(
                C_OrtDevice.cuda(), C_OrtDevice.default_memory(), idx)
        raise ValueError(
            "Unable to interpret string %r as a device." % device)
    raise TypeError(
        "Unable to interpret type %r, (%r) as de device." % (
            type(device), device))


def device_to_providers(device):
    """
    Returns the corresponding providers for a specific device.

    :param device: :epkg:`C_OrtDevice`
    :return: providers
    """
    if isinstance(device, str):
        device = get_ort_device(device)
    if device.device_type() == device.cpu():
        return ['CPUExecutionProvider']
    if device.device_type() == device.cuda():
        return ['CUDAExecutionProvider']
    raise ValueError(  # pragma: no cover
        "Unexpected device %r." % device)
