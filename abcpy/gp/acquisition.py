import numpy as np
from scipy.stats import truncnorm

from abcpy.gp.utils import stochastic_optimization, approx_second_partial_derivative, sum_of_rbf_kernels


class AcquisitionBase():
    """ All acquisition functions are assumed to fulfill this interface """

    def __init__(self, model, bounds=None):
        self.model = model
        self.bounds = bounds or [(0,1)] * model.input_dim
        if len(self.bounds) != self.model.input_dim:
            raise ValueError("Bounds dimensionality doesn't match with the model.input_dim")


    def _eval(self, x):
        """
            Evaluates the acquisition function value at 'x'

            Parameters
            ----------
            x : numpy.array
        """
        raise NotImplementedError

    def acquire(self, n_values, **kwargs):
        """
            Returns
            -------
            2d numpy array of acquisition points.
        """
        raise NotImplementedError


class LcbAcquisition(AcquisitionBase):

    def __init__(self, model, bounds=None, exploration_rate=2.0, opt_iterations=100):
        self.exploration_rate = float(exploration_rate)
        self.opt_iterations = int(opt_iterations)
        super(LcbAcquisition, self).__init__(model, bounds)

    def _eval(self, x):
        """ Lower confidence bound = mean - k * std """
        m, s2 = self.model.evaluate(x)
        return float(m - self.exploration_rate * np.sqrt(s2))

    def acquire(self, n_values, **kwargs):
        x_min, val = stochastic_optimization(self._eval, self.bounds, self.opt_iterations)
        return np.vstack([x_min]*n_values)


class BolfiAcquisition(LcbAcquisition):
    pass