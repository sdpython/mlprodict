
.. blogpost::
    :title: x / y != x * (1 / y)
    :keywords: scikit-learn, float inverse, compilation, StandardScaler
    :date: 2020-06-09
    :categories: runtime

    I was recently investigating issue
    `onnxruntime/4130 <https://github.com/microsoft/onnxruntime/issues/4130>`_
    in notebook :ref:`onnxdiscrepenciesrst`.
    While looking into a way to solve it, I finally discovered
    that this is not an easy problem.

    * `Division algorithm
      <https://en.wikipedia.org/wiki/Division_algorithm>`_
    * `Efficient floating-point division with constant integer divisors
      <https://stackoverflow.com/questions/35527683/efficient-floating-point-division-with-constant-integer-divisors>`_
    * `Will the compiler optimize division into multiplication
      <https://stackoverflow.com/questions/35506226/will-the-compiler-optimize-division-into-multiplication>`_
    * `Accelerating Correctly Rounded Floating-Point Division When the Divisor is Known in Advance
      <http://perso.ens-lyon.fr/nicolas.brisebarre/Publi/fpdivision.pdf>`_
