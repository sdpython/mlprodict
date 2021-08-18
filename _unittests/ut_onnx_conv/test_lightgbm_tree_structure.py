"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import copy
import json
import base64
import lzma
import numpy
from pandas import DataFrame
from pyquickhelper.pycode import ExtTestCase

try:
    from pyquickhelper.pycode.unittest_cst import decompress_cst
except ImportError:
    decompress_cst = lambda d: json.loads(lzma.decompress(base64.b64decode(b"".join(d))))

from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from mlprodict.onnx_conv.helpers.lgbm_helper import (
    modify_tree_for_rule_in_set, restore_lgbm_info)
from mlprodict.onnx_conv.parsers.parse_lightgbm import MockWrappedLightGbmBoosterClassifier
from mlprodict.onnx_conv import register_converters, to_onnx
from mlprodict.onnxrt import OnnxInference


def count_nodes(tree, done=None):
    if done is None:
        done = {}
    tid = id(tree)
    if tid in done:
        return 0
    done[tid] = tree
    nb = 1
    if 'right_child' in tree:
        nb += count_nodes(tree['right_child'], done)
    if 'left_child' in tree:
        nb += count_nodes(tree['left_child'], done)
    return nb


def clean_tree(tree):
    def walk_through(tree):
        if 'tree_structure' in tree:
            for w in walk_through(tree['tree_structure']):
                yield w
        yield tree
        if 'left_child' in tree:
            for w in walk_through(tree['left_child']):
                yield w
        if 'right_child' in tree:
            for w in walk_through(tree['right_child']):
                yield w

    nodes = list(walk_through(tree3))
    if True:
        for node in nodes:
            for k in ['split_gain', 'split_feature', 'split_index', 'leaf_count',
                      'internal_value', 'internal_weight', 'internal_count', 'leaf_weight']:
                if k in node:
                    del node[k]
            for k in ['leaf_value', 'leaf_value']:
                if k in node:
                    node[k] = 0


tree2 = {'average_output': False,
         'feature_names': ['c1', 'c2', 'c3', 'c4'],
         'label_index': 0,
         'max_feature_idx': 3,
         'name': 'tree',
         'num_class': 1,
         'num_tree_per_iteration': 1,
         'objective': 'binary sigmoid:1',
         'pandas_categorical': None,
         'tree_info': [{'num_cat': 0,
                        'num_leaves': 6,
                        'shrinkage': 1,
                        'tree_index': 0,
                        'tree_structure': {'decision_type': '==',
                                           'default_left': True,
                                           'internal_count': 1612,
                                           'internal_value': 0,
                                           'left_child': {'decision_type': '<=',
                                                          'default_left': True,
                                                          'internal_count': 1367,
                                                          'internal_value': 1.02414,
                                                          'left_child': {'decision_type': '<=',
                                                                         'default_left': True,
                                                                         'internal_count': 623,
                                                                         'internal_value': 1.02414,
                                                                         'left_child': {'leaf_count': 253,
                                                                                        'leaf_index': 0,
                                                                                        'leaf_value': 3.7749963852295396},
                                                                         'missing_type': 'None',
                                                                         'right_child': {'leaf_count': 370,
                                                                                         'leaf_index': 5,
                                                                                         'leaf_value': 3.7749963852295396},
                                                                         'split_feature': 3,
                                                                         'split_gain': 1.7763600157738027e-15,
                                                                         'split_index': 4,
                                                                         'threshold': 3.5000000000000004},
                                                          'missing_type': 'None',
                                                          'right_child': {'decision_type': '<=',
                                                                          'default_left': True,
                                                                          'internal_count': 744,
                                                                          'internal_value': 1.02414,
                                                                          'left_child': {'leaf_count': 291,
                                                                                         'leaf_index': 3,
                                                                                         'leaf_value': 3.7749963852295396},
                                                                          'missing_type': 'None',
                                                                          'right_child': {'leaf_count': 453,
                                                                                          'leaf_index': 4,
                                                                                          'leaf_value': 3.7749963852295396},
                                                                          'split_feature': 3,
                                                                          'split_gain': 3.552710078910475e-15,
                                                                          'split_index': 3,
                                                                          'threshold': 3.5000000000000004},
                                                          'split_feature': 2,
                                                          'split_gain': 7.105429898699844e-15,
                                                          'split_index': 2,
                                                          'threshold': 3.5000000000000004},
                                           'missing_type': 'None',
                                           'right_child': {'decision_type': '<=',
                                                           'default_left': True,
                                                           'internal_count': 245,
                                                           'internal_value': -5.7143,
                                                           'left_child': {'leaf_count': 128,
                                                                          'leaf_index': 1,
                                                                          'leaf_value': 3.130106784685405},
                                                           'missing_type': 'None',
                                                           'right_child': {'leaf_count': 117,
                                                                           'leaf_index': 2,
                                                                           'leaf_value': 3.7749963852295396},
                                                           'split_feature': 3,
                                                           'split_gain': 234.05499267578125,
                                                           'split_index': 1,
                                                           'threshold': 6.500000000000001},
                                           'split_feature': 2,
                                           'split_gain': 217.14300537109375,
                                           'split_index': 0,
                                           'threshold': '8||9||10'}},
                       {'num_cat': 0,
                        'num_leaves': 3,
                        'shrinkage': 0.05,
                        'tree_index': 1,
                        'tree_structure': {'decision_type': '<=',
                                           'default_left': True,
                                           'internal_count': 1612,
                                           'internal_value': 0,
                                           'left_child': {'leaf_count': 1367,
                                                          'leaf_index': 0,
                                                          'leaf_value': 0.05114685710677944},
                                           'missing_type': 'None',
                                           'right_child': {'decision_type': '<=',
                                                           'default_left': True,
                                                           'internal_count': 245,
                                                           'internal_value': -3.89759,
                                                           'left_child': {'leaf_count': 128,
                                                                          'leaf_index': 1,
                                                                          'leaf_value': -0.3177225912983217},
                                                           'missing_type': 'None',
                                                           'right_child': {'leaf_count': 117,
                                                                           'leaf_index': 2,
                                                                           'leaf_value': 0.05114685710677942},
                                                           'split_feature': 3,
                                                           'split_gain': 93.09839630126953,
                                                           'split_index': 1,
                                                           'threshold': 6.500000000000001},
                                           'split_feature': 2,
                                           'split_gain': 148.33299255371094,
                                           'split_index': 0,
                                           'threshold': 8.500000000000002}}],
         'version': 'v2'}


# This constant by built by appling function pyquickhelper.pycode.unittest_cst.compress_cst.

tree3 = decompress_cst([
    b'/Td6WFoAAATm1rRGAgAhARYAAAB0L+Wj4Ck9A2tdAD2IiodjqVNsvcJJI6C9h2Y0CbG5b7',
    b'OaqsqxvLBzg7BltxogYoUzxj35qbUETbBAyJeMccezEDeIKOT1GB+I50txUuc8zkWDcp/n',
    b'kx2YhORZxAyj55pXJF/xW5aySLknuTn/5cRfSL9AGF7dHdW9k8RqP5GONWx3YvvnP0tCW0',
    b'lGKd5caxoNFaB5tg+je6f0s6N6QQo8wqrBPtjJ7bQf50vFrpYgkQNAEZIVutpzgE9c4o1L',
    b'Uv/vJgnhQXOpk/4hOCV2q8VG+jD9oIjPINOOZ642k2QmsdWC+l3XagJnbN9dqT/4C9ehfM',
    b'nf6Bw5XcRXD4rtmOyUq/ocuh1WfPinlKd/Jn0YOydq1FpH+VNSUjjPUGJbJal4Pa6jcx/Y',
    b'9mcBjp9kP1pM5wkCJ52Kv12UQ/+2j+n0rUQbbqs10iFJo4h4KB/Ie/bugBLNItmNhNhDP4',
    b'36Q6jCsLsXlu0gTiWZfGQapR+DJIsVKHh9GeagotXpHTwYX72KrTFwIdxgf9Y2X1EUqiJV',
    b'wXdP7GprCs9QsIvCkqW59hPNStt2tyWtlSsXsnjU5e0Jn3USVHOcbwCBSpCtFlpg8tiS9m',
    b'Zv1TIGj9cvEk1Ke9p6bZelvtXqHJRISJ8fCVjrqTnEjyUdPaG1wmqCyz7NFEkngrBinY7e',
    b'ZMHmO1y6IhLI1zN0kq8zBHIQeqUruYgBatPI6jI585wQ6mYCobgQc7B6Ae6XlgOthATrr2',
    b'oDdnIeAPeUKVMXPIq9NnwlwsyNEoTddI42NiMde8jVzVm4wwwnqrmbKlJsi5LJhRQlaEFX',
    b'etzNn7llkCSwv88gYhcaDWP3Ewchse2iQDkJ0dPZhx0FB18X6wvEcwkt/H+dzTgAYOCSkr',
    b'T3thNkPCvQ4keiRzHiWNzLc+NAhz5NX8BXsVQFkEyf4oUkKHjy053LBmXpHM75LBhdJmFH',
    b'vqRENHF6QgiPLAjc/1NHatYLcY0VRetr55Bp2jWU+z75P2TrMkTHFnjbOEQ3p13USzVmnq',
    b'3d0EUvp5Q5dUPDFAIhkH+oUkgK4lX2xlyEGh+23EqQtmkjOyKj7HPHoPZo2AjASlRTc78u',
    b'1c9nWkTbwBGbZUsMmWzyjbDe/h2Yi2GvkSkIh8UKtYDlTzpT62G9Chf5N9HEfFjQWcdCEi',
    b'7Y3Hx86ee03jpP42ssAADRqUIMvx3yYwABhwe+UgAA2u9V4LHEZ/sCAAAAAARZWg=='])


class TestLightGbmTreeStructure(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    def test_onnxrt_python_lightgbm_categorical(self):
        val = {'decision_type': '==',
               'default_left': True,
               'internal_count': 6805,
               'internal_value': 0.117558,
               'left_child': {'leaf_count': 4293,
                              'leaf_index': 18,
                              'leaf_value': 0.003519117642745049},
               'missing_type': 'None',
               'right_child': {'leaf_count': 2512,
                               'leaf_index': 25,
                               'leaf_value': 0.012305307958365394},
               'split_feature': 24,
               'split_gain': 12.233599662780762,
               'split_index': 24,
               'threshold': '10||12||13'}

        exp = {'decision_type': '==',
               'default_left': True,
               'internal_count': 6805,
               'internal_value': 0.117558,
               'left_child': {'leaf_count': 4293,
                                'leaf_index': 18,
                                'leaf_value': 0.003519117642745049},
               'missing_type': 'None',
               'right_child': {'decision_type': '==',
                               'default_left': True,
                               'internal_count': 6805,
                               'internal_value': 0.117558,
                               'left_child': {'leaf_count': 4293,
                                                'leaf_index': 18,
                                                'leaf_value': 0.003519117642745049},
                               'missing_type': 'None',
                               'right_child': {'decision_type': '==',
                                               'default_left': True,
                                               'internal_count': 6805,
                                               'internal_value': 0.117558,
                                               'left_child': {'leaf_count': 4293,
                                                                'leaf_index': 18,
                                                                'leaf_value': 0.003519117642745049},
                                               'missing_type': 'None',
                                               'right_child': {'leaf_count': 2512,
                                                               'leaf_index': 25,
                                                               'leaf_value': 0.012305307958365394},
                                               'split_feature': 24,
                                               'split_gain': 12.233599662780762,
                                               'split_index': 24,
                                               'threshold': 13},
                               'split_feature': 24,
                               'split_gain': 12.233599662780762,
                               'split_index': 24,
                               'threshold': 12},
               'split_feature': 24,
               'split_gain': 12.233599662780762,
               'split_index': 24,
               'threshold': 10}

        nb1 = count_nodes(val)
        modify_tree_for_rule_in_set(val)
        nb2 = count_nodes(val)
        self.assertEqual(nb1, 3)
        self.assertEqual(nb2, 5)
        sval = str(val)
        self.assertNotIn('||', sval)
        self.maxDiff = None
        self.assertEqual(exp, val)

    def test_onnxrt_python_lightgbm_categorical2(self):
        val = copy.deepcopy(tree2)
        nb1 = sum(count_nodes(t['tree_structure']) for t in val['tree_info'])
        modify_tree_for_rule_in_set(val)
        nb2 = sum(count_nodes(t['tree_structure']) for t in val['tree_info'])
        self.assertEqual(nb1, 16)
        self.assertEqual(nb2, 18)

    def test_mock_lightgbm(self):
        tree = copy.deepcopy(tree2)
        nb1 = sum(count_nodes(t['tree_structure']) for t in tree['tree_info'])
        model = MockWrappedLightGbmBoosterClassifier(tree)
        nb2 = sum(count_nodes(t['tree_structure']) for t in tree['tree_info'])
        self.assertEqual(nb1, nb2)
        self.assertEqual(nb1, 16)
        onx = to_onnx(model, initial_types=[('x', FloatTensorType([None, 4]))],
                      options={id(model): {'zipmap': False}})
        self.assertTrue(model.visited)

        for n in onx.graph.node:
            if n.op_type != 'TreeEnsembleClassifier':
                continue
            att = n.attribute
            for k in att:
                if k.name != 'nodes_modes':
                    continue
                values = k.strings
                nbnodes = len(values)
        self.assertEqual(nbnodes, 18)

        iris = load_iris()
        X = iris.data
        X = (X * 10).astype(numpy.int32)

        oif = OnnxInference(onx)
        for row in [1, 10, 20, 30, 40, 50, 60, 70]:
            with self.subTest(row=row):
                pred = oif.run({'x': X[:row].astype(numpy.float32)})
                label = pred["output_label"]
                self.assertEqual(label.shape, (row, ))
                prob = DataFrame(pred["output_probability"]).values
                self.assertEqual(prob.shape, (row, 2))

    def test_mock_lightgbm_info(self):
        tree = copy.deepcopy(tree3)
        info = restore_lgbm_info(tree)
        modify_tree_for_rule_in_set(tree, info=info)
        expected = tree
        tree = copy.deepcopy(tree3)
        info = restore_lgbm_info(tree)
        modify_tree_for_rule_in_set(tree, info=info)
        self.assertEqual(expected, tree)

    def test_mock_lightgbm_profile(self):
        tree = copy.deepcopy(tree3)
        info = restore_lgbm_info(tree)
        self.assertIsInstance(info, list)
        self.assertGreater(len(info), 1)

        def g():
            for i in range(0, 100):
                modify_tree_for_rule_in_set(tree, info=info)
        p2 = self.profile(g)[1]
        self.assertIn('cumtime', p2)
        if __name__ == "__main__":
            print(p2)


if __name__ == "__main__":
    unittest.main()
