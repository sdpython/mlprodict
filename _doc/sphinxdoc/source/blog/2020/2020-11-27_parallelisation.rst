
.. blogpost::
    :title: Parallelization of Random Forest predictions
    :keywords: scikit-learn, parallelization, Random Forest
    :date: 2020-11-27
    :categories: runtime

    I've been struggling to understand why the first implementation
    of TreeEnsemble could not get as fast as *scikit-learn* implementation
    for a RandomForest when the number of observations was 100.000 or above,
    100 trees and a depth >= 10. The only difference was that the computation
    was parallelized by tree and not by observations.

    * `forest.py
      <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_forest.py#L683>`_
    * `tree.pyx
      <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx#L926>`_

    Parallelize by tree requires to save the output of every observation,
    that means the computation requires an additional buffer
    (one per thread at least) to save the trees outputs.
    However, that approximatively two, three times faster to do it
    that way instead of parallelizing per observations.
    The computational is the same in both cases. The only
    explanation would be a better use of the caches (L1, L2, L3)
    when the computation is parallelized per tree.
    The answer is probably hidden in that book.

    * `What Every Programmer Should Know About Memory
      <https://akkadia.org/drepper/cpumemory.pdf>`_
