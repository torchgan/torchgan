__all__ = ['EvaluationMetric']


class EvaluationMetric(object):
    def preprocess(self, x):
        pass

    def calculate_score(self, x):
        pass

    def __call__(self, x):
        return self.calculate_score(self.preprocess(x))
