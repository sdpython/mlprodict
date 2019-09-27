"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_conv.operator_converters.conv_lightgbm import modify_tree_for_rule_in_set


class TestLightGbmTreeStructur(ExtTestCase):

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

        modify_tree_for_rule_in_set(val)
        sval = str(val)
        self.assertNotIn('||', sval)
        self.maxDiff = None
        self.assertEqual(exp, val)


if __name__ == "__main__":
    unittest.main()
