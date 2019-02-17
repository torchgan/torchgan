__all__ = ["EvaluationMetric"]


class EvaluationMetric(object):
    r"""
    Base class for all Evaluation Metrics
    """

    def __init__(self):
        self.arg_map = {}

    def set_arg_map(self, value):
        r"""Updates the ``arg_map`` for passing a different value to the ``metric_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``metric_ops`` signature is
            ``metric_ops(self, gen, disc)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``metric.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def preprocess(self, x):
        r"""
        Subclasses must override this function and provide their own preprocessing
        pipeline.

        :raises NotImplementedError: If the subclass doesn't override this function.
        """
        raise NotImplementedError

    def calculate_score(self, x):
        r"""
        Subclasses must override this function and provide their own score calculation.

        :raises NotImplementedError: If the subclass doesn't override this function.
        """
        raise NotImplementedError

    def metric_ops(self, generator, discriminator, **kwargs):
        r"""
        Subclasses must override this function and provide their own metric evaluation ops.

        :raises NotImplementedError: If the subclass doesn't override this function.
        """
        raise NotImplementedError

    def __call__(self, x):
        return self.calculate_score(self.preprocess(x))
