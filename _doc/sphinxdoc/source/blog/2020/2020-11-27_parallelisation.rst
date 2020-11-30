
.. blogpost::
    :title: Parallelization of Random Forest predictions
    :keywords: scikit-learn, parallelization, Random Forest
    :date: 2020-11-27
    :categories: runtime

    I've been struggling to understand why the first implementation
    of TreeEnsemble could not get as fast as *scikit-learn* implementation
    for a RandomForest when the number of observations was 100.000 or above,
    100 trees and a depth >= 10. The only difference was that the computation
    was parallelized by trees and not by observations. These
    observations are benchmarked in
    :ref:`l-example-tree-ensemble-reg-bench`
    (:ref:`l-example-tree-ensemble-cls-bench-multi` for the
    multiclass version).

    * `forest.py
      <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_forest.py#L683>`_
    * `tree.pyx
      <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx#L926>`_

    Parallelizing by tree requires to save the outputs of every observation.
    That means the computation requires an additional buffer
    (one per thread at least) to save the trees outputs.
    However, that approximatively two, three times faster to do it
    that way instead of parallelizing per observations.
    The computational is the same in both cases. The only
    explanation would be a better use of the caches (L1, L2, L3)
    when the computation is parallelized per tree.
    The answer is probably hidden in that book.

    * `What Every Programmer Should Know About Memory
      <https://akkadia.org/drepper/cpumemory.pdf>`_

    The next investigation should be a study of the difference
    between a tree described as an array of nodes or
    a structure of arrays where every node field gets its own array.

    * `Performance Optimization Strategies for WRF Physics Schemes
      Used in Weather Modeling
      <https://www.researchgate.net/figure/
      Transformation-from-AOS-to-SOA-The-2D-arrays-A-and-B-are-transformed-into-two_fig1_326307125>`_
    * `Memory Layout Transformations
      <https://software.intel.com/content/www/us/en/develop/articles/
      memory-layout-transformations.html>`_

    Other readings:

    * `Demystifying The Restrict Keyword
      <https://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html>`_
    * `Aliasing <https://en.wikipedia.org/wiki/Aliasing_%28computing%29>`_
