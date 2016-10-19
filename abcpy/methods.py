import numpy as np
import dask
from distributed import Client

from abcpy.gp.acquisition import BolfiAcquisition
from abcpy.gp.gpy_model import GPyModel
from .async import wait


class ABCMethod(object):
    def __init__(self, N, distance_node=None, parameter_nodes=None, batch_size=10):

        if not distance_node or not parameter_nodes:
            raise ValueError("Need to give the distance node and list of parameter nodes")

        self.N = N
        self.distance_node = distance_node
        self.parameter_nodes = parameter_nodes
        self.batch_size = batch_size

    def infer(self, threshold, *args, **kwargs):
        raise NotImplementedError


class Rejection(ABCMethod):
    """
    Rejection sampler.
    """
    def infer(self, threshold, *args, **kwargs):
        """
        Run the rejection sampler. Inference can be repeated with a different
        threshold without rerunning the simulator.
        """

        # only run at first call
        if not hasattr(self, 'distances'):
            self.distances = self.distance_node.generate(self.N, batch_size=self.batch_size).compute()
            self.parameters = [p.acquire(self.N).compute() for p in self.parameter_nodes]

        accepted = self.distances < threshold
        posteriors = [p[accepted] for p in self.parameters]

        return posteriors


class BOLFI(ABCMethod):

    def __init__(self, N, distance_node=None, parameter_nodes=None, batch_size=2, model=None, acquisition=None, n_surrogate_samples=None):
        self.model = model or GPyModel(len(parameter_nodes))
        self.acquisition = acquisition or BolfiAcquisition(self.model)
        self.n_surrogate_samples = n_surrogate_samples or 20
        super(BOLFI, self).__init__(N, distance_node, parameter_nodes, batch_size)

    def infer(self, threshold=None, *args, **kwargs):
        """
            Bolfi inference.

            type(threshold) = float
        """
        self.create_surrogate_likelihood()
        return self.sample_posterior(threshold)

    def create_surrogate_likelihood(self, n_surrogate_samples=None):
        n_surrogate_samples = n_surrogate_samples or self.n_surrogate_samples
        while self.model.n_observations() < n_surrogate_samples:
            x = self.acquisition.acquire(self.batch_size)
            param_values = {p.name: x[:,i,None] for i, p in enumerate(self.parameter_nodes)}
            y = self.distance_node.generate(len(x), with_values=param_values).compute()
            self.model.update(x, y)
        return self.model

    def getPosterior(self, threshold):
        raise NotImplementedError()

    def sample_posterior(self, threshold):
        raise NotImplementedError()
