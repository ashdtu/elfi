import numpy as np
import GPy


class GPyModel:

    # Defaults
    kernel_class = GPy.kern.RBF
    kernel_var = 1.0
    kernel_lengthscale = 1.0
    noise_var = 1.0
    # optimizers: lbfgsb, simplex, scg, adadelta, rasmussen
    optimizer = "lbfgsb"
    opt_max_iters = int(1e5)

    def __init__(self, input_dim, kernel=None):
        self.input_dim = input_dim
        self.kernel = kernel or self._get_kernel()
        self.gp = None

    def evaluate(self, x):
        """ Returns the mean, variance of the GP at x as floats """
        if self.gp is None:
            return 0.0, 0.0
        m, s2 = self.gp.predict(np.atleast_2d(x))
        return m, s2

    def eval_mean(self, x):
        m, s2 = self.evaluate(x)
        return m

    def update(self, X, Y):
        if self.gp is None:
            gp = self._get_gp(X, Y)
        else:
            X = np.vstack((self.gp.X, X))
            Y = np.vstack((self.gp.Y, Y))
            # This doesn't update self.gp.num_data, a bug?
            #self.gp.set_XY(X, Y)
            gp = self._get_gp(X, Y)


        gp.optimize(self.optimizer, max_iters=self.opt_max_iters)
        self.gp = gp
        #except np.linalg.linalg.LinAlgError as e:
        #    print("Numerical error in GP optimization! Let's hope everything still works.")

    def n_observations(self):
        """ Returns the number of observed samples """
        if self.gp is None:
            return 0
        return self.gp.num_data

    def _get_kernel(self):
        """ Internal function to create a kernel for GPy model
        Available at least: exponential, expquad, matern32, matern52
        """

        if isinstance(self.kernel_class, str):
            self.kernel_class = getattr(GPy.kern, self.kernel_class)
        # noinspection PyCallingNonCallable
        return self.kernel_class(input_dim=self.input_dim,
                                 variance=self.kernel_var,
                                 lengthscale=self.kernel_lengthscale)

    def _get_gp(self, X, Y):
        gp = GPy.models.GPRegression(X=X,
                                     Y=Y,
                                     kernel=self.kernel,
                                     noise_var=self.noise_var)
        return gp