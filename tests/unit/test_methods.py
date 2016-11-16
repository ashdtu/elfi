import numpy as np
from distributed import Client
from functools import partial
import GPy
import elfi
from elfi.methods import _SMC_Distribution


# Test case
class MockModel():

    def mock_simulator(self, p, n_sim=1, prng=None):
        self.mock_sim_calls += np.atleast_2d(p).shape[0]
        print(self.mock_sim_calls, np.atleast_2d(p).shape[0])
        return np.hstack([p, p])

    def mock_summary(self, x):
        self.mock_sum_calls += x.shape[0]
        m = np.mean(x, axis=1, keepdims=True)
        return m

    def mock_discrepancy(self, x, y):
        self.mock_dis_calls += x[0].shape[0]
        d = np.linalg.norm(np.array(x) - np.array(y), axis=0, ord=1)
        return d

    def set_simple_model(self, vectorized=True):
        self.mock_sim_calls = 0
        self.mock_sum_calls = 0
        self.mock_dis_calls = 0
        self.bounds = ((0, 1),)
        self.input_dim = 1
        self.obs = self.mock_simulator(0.)
        self.mock_sim_calls = 0
        self.p = elfi.Prior('p', 'uniform', 0, 1)
        self.Y = elfi.Simulator('Y', self.mock_simulator, self.p,
                                observed=self.obs, vectorized=vectorized)
        self.S = elfi.Summary('S', self.mock_summary, self.Y)
        self.d = elfi.Discrepancy('d', self.mock_discrepancy, self.S)


# Tests for the base class
class Test_ABCMethod(MockModel):

    def test_constructor(self):
        p1 = elfi.Prior('p1', 'uniform', 0, 1)
        p2 = elfi.Prior('p2', 'uniform', 0, 1)
        d = elfi.Discrepancy('d', np.mean, p1, p2)
        abc = elfi.ABCMethod(d, [p1, p2])

        try:
            abc = elfi.ABCMethod()
            abc = elfi.ABCMethod(0.2, None)
            abc = elfi.ABCMethod([d], [p1, p2])
            abc = elfi.ABCMethod(d, p1)
            assert False
        except:
            assert True

    def test_sample(self):
        p1 = elfi.Prior('p1', 'uniform', 0, 1)
        d = elfi.Discrepancy('d', np.mean, p1)
        abc = elfi.ABCMethod(d, [p1])
        try:
            abc.sample()  # NotImplementedError
            assert False
        except:
            assert True

    def test_get_distances(self):
        self.set_simple_model()
        abc = elfi.ABCMethod(self.d, [self.p], batch_size=1)
        n_sim = 4
        distances, parameters = abc._get_distances(n_sim)
        print(distances)
        assert distances.shape == (n_sim, 1)
        assert isinstance(parameters, list)
        assert parameters[0].shape == (n_sim, 1)


# Tests for rejection sampling
class Test_Rejection(MockModel):

    def test_quantile(self):
        self.set_simple_model()

        n = 20
        batch_size = 10
        quantile = 0.5
        rej = elfi.Rejection(self.d, [self.p], batch_size=batch_size)

        result = rej.sample(n, quantile=quantile)
        assert isinstance(result, dict)
        assert 'samples' in result.keys()
        assert result['samples'][0].shape == (n, 1)
        assert self.mock_sim_calls == int(n / quantile)
        assert self.mock_sum_calls == int(n / quantile) + 1
        assert self.mock_dis_calls == int(n / quantile)

    def test_threshold(self):
        self.set_simple_model()

        n = 20
        batch_size = 10
        rej = elfi.Rejection(self.d, [self.p], batch_size=batch_size)
        threshold = 0.1

        result = rej.sample(n, threshold=threshold)
        assert isinstance(result, dict)
        assert 'samples' in result.keys()
        assert self.mock_sim_calls == int(n)
        assert self.mock_sum_calls == int(n) + 1
        assert self.mock_dis_calls == int(n)
        assert np.all(result['samples'][0] < threshold)  # makes sense only for MockModel!

    def test_distributed_threshold(self):  # uses Dask.Distributed.Client with LocalCluster
        elfi.env.client()
        self.set_simple_model()

        n = 40
        batch_size = 10
        rej = elfi.Rejection(self.d, [self.p], batch_size=batch_size)
        threshold = 0.1

        result = rej.sample(n, threshold=threshold)
        assert isinstance(result, dict)
        assert 'samples' in result.keys()
        assert np.all(result['samples'][0] < threshold)  # makes sense only for MockModel!
        elfi.env.client().shutdown()


class Test_SMC(MockModel):

    def test_SMC_dist(self):
        current_params = np.array([1., 10., 100., 1000.])[:, None]
        weighted_sd = np.array([1.])
        weights = np.array([0., 0., 1., 0.])
        weights /= np.sum(weights)
        random_state = np.random.RandomState(0)
        params = _SMC_Distribution.rvs(current_params, weighted_sd, weights, random_state, size=current_params.shape)
        assert params.shape == (4, 1)
        assert np.allclose(params, current_params[2, 0], atol=5.)
        p = _SMC_Distribution.pdf(params, current_params, weighted_sd, weights)
        assert p.shape == (4, 1)

    def test_SMC(self):
        self.set_simple_model()

        n = 20
        batch_size = 10
        smc = elfi.SMC(self.d, [self.p], batch_size=batch_size)
        n_populations = 3
        schedule = [0.5] * n_populations

        prior_id = id(self.p)
        result = smc.sample(n, n_populations, schedule)

        assert id(self.p) == prior_id  # changed within SMC, finally reverted
        assert self.mock_sim_calls == int(n / schedule[0] * n_populations)
        assert self.mock_sum_calls == int(n / schedule[0] * n_populations) + 1
        assert self.mock_dis_calls == int(n / schedule[0] * n_populations)

    def test_distributed_SMC(self):  # uses Dask.Distributed.Client with LocalCluster
        elfi.env.client()
        self.set_simple_model()

        n = 40
        batch_size = 10
        smc = elfi.SMC(self.d, [self.p], batch_size=batch_size)
        n_populations = 3
        schedule = [0.5] * n_populations

        prior_id = id(self.p)
        result = smc.sample(n, n_populations, schedule)

        assert id(self.p) == prior_id  # changed within SMC, finally reverted
        elfi.env.client().shutdown()


class Test_BOLFI(MockModel):

    def set_basic_bolfi(self):
        self.n_sim = 2
        self.n_batch = 1
        self.kernel_class = "Matern32"
        self.kernel_var = 1.0
        self.kernel_scale = 1.0
        self.model = elfi.GPyModel(input_dim=self.input_dim,
                              bounds=self.bounds,
                              kernel_class=self.kernel_class,
                              kernel_var=self.kernel_var,
                              kernel_scale=self.kernel_scale)
        self.acq = elfi.BolfiAcquisition(self.model, n_samples=self.n_sim,
                                    exploration_rate=2.5, opt_iterations=1000)

    def test_basic_sync_use(self):
        self.set_simple_model(vectorized=False)
        self.set_basic_bolfi()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           model=self.model, acquisition=self.acq, sync=True)
        post = bolfi.infer()
        assert self.acq.finished is True
        assert bolfi.model.n_observations == self.n_sim

    def test_basic_async_use(self):
        self.set_simple_model(vectorized=False)
        self.set_basic_bolfi()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           model=self.model, acquisition=self.acq, sync=False)
        post = bolfi.infer()
        assert self.acq.finished is True
        assert bolfi.model.n_observations == self.n_sim
