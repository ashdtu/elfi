import numpy as np
from scipy.stats import truncnorm

from abcpy.gp.utils import stochastic_optimization, approx_second_partial_derivative, sum_of_rbf_kernels


class AcquisitionBase():
    """ All acquisition functions are assumed to fulfill this interface """

    def __init__(self, model, bounds=None):
        self.model = model
        self.bounds = bounds or [(0,1)] * model.input_dim

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

    def acquire(self, n_values, pending_locations=None):
        x_min, val = stochastic_optimization(self._eval, self.model.bounds, self.opt_iterations)
        return np.vstack([x_min]*n_values)


class SecondDerivativeNoiseMixin(AcquisitionBase):

    def __init__(self, *args, **kwargs):
        self.second_derivative_delta = float(kwargs.get("second_derivative_delta", 0.01))
        super(SecondDerivativeNoiseMixin, self).__init__(*args, **kwargs)

    def acquire(self, n_values, pending_locations=None):
        """ Adds noise based on function second derivative """
        x = super(SecondDerivativeNoiseMixin, self).acquire(n_values)
        locs = list()
        for i in range(n_values):
            xi = x[i,:]
            loc = list()
            for dim, val in enumerate(xi.tolist()):
                d2 = approx_second_partial_derivative(self._eval, xi, dim,
                        self.second_derivative_delta, self.bounds)
                # std from mathching second derivative to that of normal
                # -N(0,std)'' = 1/(sqrt(2pi)std^3) = der2
                # => std = der2 ** -1/3 * (2*pi) ** -1/6
                if d2 > 0:
                    std = np.power(2*np.pi, -1.0/6.0) * np.power(d2, -1.0/3.0)
                else:
                    std = float("inf")
                low = self.bounds[i][0]
                high = self.bounds[i][1]
                maxstd = (high - low) / 2.0
                std = min(std, maxstd)  # limit noise amount based on bounds
                newval = truncnorm.rvs(low, high, loc=val, scale=std)
                loc.append(newval)
            locs.append(loc)
        return np.atleast_2d(locs)


class BolfiAcquisition(SecondDerivativeNoiseMixin, LcbAcquisition):
    pass