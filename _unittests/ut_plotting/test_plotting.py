# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import os
import platform
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.plotting.plotting import plot_benchmark_metrics


class TestPlotBenchScatter(ExtTestCase):

    @unittest.skipIf(platform.platform() == 'win32' and __name__ != '__main__',
                     reason="Message: 'generated new fontManager'")
    def test_plot_logreg_xtime(self):
        temp = get_temp_folder(__file__, "temp_plot_benchmark_metrics")
        img = os.path.join(temp, "plot_bench.png")

        data = {(1, 1): 0.1, (10, 1): 1, (1, 10): 2,
                (10, 10): 100, (100, 1): 100, (100, 10): 1000}
        import matplotlib
        if __name__ != "__main__":
            back = matplotlib.get_backend()
            matplotlib.use('Agg')
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise AssertionError(
                "Failure due to %r (platform=%r, __name__=%r)." % (
                    e, platform.platform(), __name__)) from e
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
