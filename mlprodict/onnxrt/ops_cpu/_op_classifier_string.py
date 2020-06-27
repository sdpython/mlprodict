"""
@file
@brief Common class for classifiers supporting strings.
"""
import numpy


class _ClassifierCommon:
    """
    Labels strings are not natively implemented in C++ runtime.
    The class stores the strings labels, replaces them by
    integer, calls the C++ codes and then replaces them by strings.
    """

    def _post_process_label_attributes(self):
        """
        Replaces string labels by int64 labels.
        It creates attributes *_classlabels_int64s_string*.
        """
        name_int = 'classlabels_int64s' if hasattr(
            self, 'classlabels_int64s') else 'classlabels_ints'
        if (hasattr(self, 'classlabels_strings') and
                len(self.classlabels_strings) > 0):  # pylint: disable=E0203
            if hasattr(self, name_int) and len(getattr(self, name_int)) != 0:
                raise RuntimeError(  # pragma: no cover
                    "'%s' must be empty if "
                    "'classlabels_strings' is not." % name_int)
            setattr(self, name_int, numpy.arange(len(self.classlabels_strings),  # pylint: disable=E0203
                                                 dtype=numpy.int64))
            self._classlabels_int64s_string = self.classlabels_strings  # pylint: disable=E0203
            self.classlabels_strings = numpy.empty(
                shape=(0, ), dtype=numpy.str)
        else:
            self._classlabels_int64s_string = None

    def _post_process_predicted_label(self, label, scores):
        """
        Replaces int64 predicted labels by the corresponding
        strings.
        """
        if self._classlabels_int64s_string is not None:
            label = numpy.array(
                [self._classlabels_int64s_string[i] for i in label])
        return label, scores
