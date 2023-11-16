import itertools
from SmoothLaplacian import SmoothLaplacian
import numpy as np

def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)

class SmoothLaplacianAuto(SmoothLaplacian):
    """Smooth precosopm w/ cross-validated choice of the l1 penalty.

    Parameters
    ----------
    alphas : int or array-like of shape (n_alphas,), dtype=float, default=4
        If an integer is given, it fixes the number of points on the
        grids of alpha to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details. Range is [1, inf) for an integer.
        Range is (0, inf] for an array-like of floats.

    betas : int or array-like of shape (n_betas,), dtype=float, default=4
        If an integer is given, it fixes the number of points on the
        grids of beta to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details. Range is [1, inf) for an integer.
        Range is (0, inf] for an array-like of floats.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    max_iter : int, default=100
        Maximum number of iterations.

    solver = string, default  = 'cvxopt'
        The solver parameter in the qpsolver parameter.
        Options are: ['clarabel', 'cvxopt', 'daqp', 'ecos', 'highs', 'osqp', 'piqp', 
        'proxqp', 'quadprog', 'scs']. For more information on the solvers, see 
        https://pypi.org/project/qpsolvers/
        Run: pip install qpsolvers[open_source_solvers] 
        to install all available solvers.

        
    Attributes
    ----------
    Laplacian : ndarray of shape (n_features, n_features)
        Estimated laplacian matrix.

    alpha : float
        Penalization parameter alpha selected.

    beta : float
        Penalization parameter beta selected.

    results : dict of ndarrays
        A dict with keys:

        parameters : ndarray of shape (n_alphas*n_betas)
            All penalization parameters explored.

        losses : ndarray of shape (n_alphas*n_betas))
            losses of parameters.
    
    Y : array-like of shape (n_nodes, n_nodes)
        True signals of X without noises 

    loss: float
        The value of objective function with the current Laplacian and Y

    precision : array-like of shape (n_observations, n_nodes)
        Assumed to be the Laplacian

    covariance : array-like of shape (n_observations, n_nodes)
        The Moore-Penrose pseudoinverse of Laplacian

    partial_correlations : array-like of shape (n_observations, n_nodes)
            Partial correlation matrix

    adj : array-like of shape (n_observations, n_nodes)
        Adjacency matrix according to Laplacian thresheld at threshold

    """

    def __init__(
        self,
        alphas=4,
        betas=4,
        max_iter=100,
        tol=1e-4,
        solver='cvxopt'
    ):
        super().__init__(
            tol=tol,
            max_iter=max_iter,
            solver=solver
        )
        self.alphas = alphas
        self.betas = betas

    def fit(self, X):
        """Fit the SmoothLaplacian model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        n_alphas = self.alphas
        if _is_arraylike_not_scalar(n_alphas):
            alphas = self.alphas
        elif isinstance(n_alphas, int):
            alphas = np.logspace(-1.5, 1, num=n_alphas)
        else:
            alphas = np.logspace(-1.5, 1, num=10)

        n_betas = self.betas
        if _is_arraylike_not_scalar(n_betas):
            betas = self.betas
        elif isinstance(n_betas, int):
            betas = np.logspace(-1.5, 1, num=n_betas)
        else:
            betas = np.logspace(-1.5, 1, num=10)
        params = list(itertools.product(alphas,betas))

        # Find the maximum
        losses = []
        self.results = {}
        for alpha,beta in params:
            model = SmoothLaplacian(alpha = alpha, beta = beta, tol=self.tol, max_iter=self.max_iter, solver=self.solver)
            model.fit(X)
            loss = model.loss
            losses.append(loss)
            self.results[(alpha,beta)] = loss
        losses = np.array(losses)
        best_index = np.argmin(losses)
        self.alpha, self.beta = params[best_index]

        # Finally fit the model with the selected alpha
        self.model = SmoothLaplacian(
            alpha=self.alpha,
            beta = self.beta,
            tol=self.tol,
            max_iter=self.max_iter,
            solver=self.solver
        )
        self.model.fit(X)
        self.Laplacian, self.Y, self.loss,self.precision,self.covariance, self.partial_correlations, self.adj = self.model.Laplacian, self.model.Y, self.model.loss, self.model.precision,self.model.covariance, self.model.partial_correlations, self.model.adj
        return self