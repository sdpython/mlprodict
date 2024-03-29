# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""
import os
import platform
import unittest
from pyquickhelper.pycode import (
    ExtTestCase, get_temp_folder, is_travis_or_appveyor)
from mlprodict.plotting.plotting import plot_benchmark_metrics


class TestPlotBenchScatter(ExtTestCase):

    @unittest.skipIf(
        platform.platform() == 'win32' and is_travis_or_appveyor() == 'azurepipe',
        reason="Message: 'generated new fontManager'")
    def test_plot_logreg_xtime(self):
        temp = get_temp_folder(__file__, "temp_plot_benchmark_metrics")
        img = os.path.join(temp, "plot_bench.png")

        data = {(1, 1): 0.1, (10, 1): 1, (1, 10): 2,
                (10, 10): 100, (100, 1): 100, (100, 10): 1000}
        import matplotlib
        if __name__ != "__main__":
            try:
                back = matplotlib.get_backend()
            except Exception as e:
                raise AssertionError(  # pylint: disable=W0707
                    "Failure (1) due to %r (platform=%r, __name__=%r)." % (
                        e, platform.platform(), __name__))
            matplotlib.use('Agg')
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise AssertionError(  # pylint: disable=W0707
                "Failure (2) due to %r (platform=%r, __name__=%r)." % (
                    e, platform.platform(), __name__))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        plot_benchmark_metrics(data, ax=ax[0], cbar_kw={'shrink': 0.6})
        plot_benchmark_metrics(data, ax=ax[1], transpose=True, xlabel='X', ylabel='Y',
                               cbarlabel="ratio")
        if __name__ == "__main__":
            fig.savefig(img)
            self.assertExists(img)
            plt.show()
        plt.close('all')
        if __name__ != "__main__":
            matplotlib.use(back)


if __name__ == "__main__":
    unittest.main()
