# from cvxopt import matrix, solvers
# solvers.options['show_progress'] = False
from qpsolvers import solve_qp
import numpy as np
from scipy import sparse
import math

def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")

def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)

class SmoothLaplacian():
    '''
    Implementation of the paper 'Learning Laplacian Matrix in 
    Smooth Graph Signal Representations'

    Objective function is:
    max_L { alpha * tr(X^T L X) - beta * Frobenius norm(L)^2 }
    to learn the Laplacian of the smoothest graph. 
    
    Here, X is the data matrix.

    Parameters
    ----------
    alpha : float, default=0.01
        The smoothness parameter: the higher alpha, the more
        focus on smoothness, the smoother the graph with respect to data.
        Range is (0, inf].

    beta : float, default=0.01
        The regularization parameter: the higher beta, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    solver = string, default  = 'cvxopt'
        The solver parameter in the qpsolver parameter for quadratic programing optimization.
        Options are: ['clarabel', 'cvxopt', 'daqp', 'ecos', 'highs', 'osqp', 'piqp', 
        'proxqp', 'quadprog', 'scs']. For more information on the solvers, see 
        https://pypi.org/project/qpsolvers/
        Run: pip install qpsolvers[open_source_solvers] 
        to install all available solvers.
    '''

    def __init__(
        self,
        alpha=0.01,
        beta=0.01,
        max_iter=100,
        tol=1e-8,
        solver='cvxopt',
        **kwargs
    ):
        self.alpha = alpha
        self.beta = beta
        self.max_iter=max_iter
        self.tol = tol
        self.solver=solver
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, threshold = None):
        '''
        Fit the SmoothLapcacian model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_nodes)
            Data from which to compute the estimated Laplacian.

        threshold : float, default = None.
            Threshold to turn Laplacian to adjacency matrix under the Gaussian
            Markov Random Field model. If None, then it is calculated as 
            threshold = sqrt(log(n_nodes) / n_observations)

        Returns
        ----------
        Laplacian : array-like of shape (n_observations, n_nodes)
            Laplacian fitted from X

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
        '''
        self.Laplacian, self.Y, self.loss = self.GL_SigRep(X,self.max_iter,self.alpha,self.beta,self.tol,self.solver)
        self.precision = self.Laplacian
        if self.Laplacian is None:
            self.covariance = None
            self.partial_correlations = None
            self.adj = None
        else: 
            self.covariance = np.linalg.pinv(self.precision.astype(float))
            d = 1 / np.sqrt(np.diag(self.precision))
            self.partial_correlations = self.precision*d*d[:, np.newaxis]
            if not threshold:
                threshold = math.sqrt(math.log(X.shape[1])/X.shape[0])
            self.adj = np.abs(np.triu(self.partial_correlations, k=1)) > threshold

    def GL_SigRep(self, X, max_iter, alpha, beta, tol, solver):
        """
        Returns Output Signal Y, Graph Laplacian L

        Parameters
        ----------
        X : array-like of shape (n_observations, n_nodes)
            Data from which to compute the estimated Laplacian.

        max_iter : int, default=100
            The maximum number of iterations.

        alpha : float, default=0.01
            The smoothness parameter: the higher alpha, the more
            focus on smoothness, the smoother the graph with respect to data.
            Range is (0, inf].

        beta : float, default=0.01
            The regularization parameter: the higher beta, the more
            regularization, the sparser the inverse covariance.
            Range is (0, inf].

        tol : float, default=1e-4
            The tolerance to declare convergence: if the dual gap goes below
            this value, iterations are stopped. Range is (0, inf].

        solver = string, default  = 'cvxopt'
            The solver parameter in the qpsolver parameter.
            Options are: ['clarabel', 'cvxopt', 'daqp', 'ecos', 'highs', 'osqp', 'piqp', 
            'proxqp', 'quadprog', 'scs']. For more information on the solvers, see 
            https://pypi.org/project/qpsolvers/
            Run: pip install qpsolvers[open_source_solvers] 
            to install all available solvers.
        """

        # initialize the Y matrix to be X.T
        Y = X.T
        # n is the number of nodes
        n = X.shape[1]

        M_dup, P, A, b, G, h = self.qp_matrices(n, beta)

        # curr_cost = np.linalg.norm(np.ones((n, n)), 'fro')
        curr_cost = float('inf')

        for i in range(max_iter):

            # Update cost function
            prev_cost = curr_cost

            # the matrix q in the linear form to be put into the qp solver
            q = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_dup)
            # Assuming Y matrix, solve the quadratic programing problem
            # in terms of L using cvxopt
            sol = solve_qp(P, q, G, h, A, b, solver=solver)
            l_vech = np.array(sol)
            l_vec = np.dot(M_dup, l_vech)
            L = l_vec.reshape(n, n)

            # Assert L is correctly learnt.
            if not np.allclose(L.trace(), n):
                # print(f'Solution does not satisfy the trace constraint for alpha, beta : ({alpha},{beta}).')
                return None, None, float('inf')
            if not np.all(L - np.diag(np.diag(L)) <= 0):
                # print(f'Solution does not satisfy the non-positive off-diagonal entries constraint for alpha, beta : ({alpha},{beta}).')
                return None, None, float('inf')
            if not np.allclose(np.dot(L, np.ones(n)), np.zeros(n)):
                # print(f'Solution does not satisfy the zero row-sum constraint for alpha, beta : ({alpha},{beta}).')
                return None, None, float('inf')
        
            # Update Y
            Y = np.dot(np.linalg.inv(np.eye(n) + alpha * L), X.T)

            # Update the current cost function
            curr_cost = (np.linalg.norm(X.T - Y, 'fro')**2 +
                        alpha * np.dot(np.dot(Y.T, L), Y).trace() +
                        beta * np.linalg.norm(L, 'fro')**2)

            if np.abs(curr_cost - prev_cost) < tol:
                break
        return L, Y, curr_cost
        
    def qp_matrices(self, n, beta):
        """
        Returns the matrix in the quadratic programing problem.

        Parameters
        ----------
        n : int,
            number of nodes

        beta : float, default=0.01
            The regularization parameter: the higher beta, the more
            regularization, the sparser the inverse covariance.
            Range is (0, inf].
        """
        # duplication matrix
        M_dup = self.duplication_matrix(n)
        # the matrix P in the quadratic form
        P = 2 * beta * np.dot(M_dup.T, M_dup)
        P = sparse.csr_matrix(P)
        # A matrix for the L * 1 = 0 constraint (zero row-sum) and tr(L) = n 
        # constraint (trace constraint)
        A = self.A_matrix(n)
        A = sparse.csr_matrix(A)
        # b constraint vector in A * vech(L) = b
        b = self.b_constraint(n)
        # G matrix for G * vech <= h, i.e., L_ij = L_ji <= 0 (non-positive 
        # off-diagonal entries).
        G = self.G_matrix(n)
        G = sparse.csr_matrix(G)
        # b vector for the G * vech <= h. h = 0.
        h = np.zeros(n*(n-1)//2)
        return M_dup, P, A, b, G, h
    
    def duplication_matrix(self,n):
        """
        Returns the duplication matrix M. M is the unique n^2 + (n * (n + 1))/2 
        matrix such that for any n x n symmetric matrix A:
        M * vech(A) = vec(A)

        https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices

        Below is a modification of the C++ code in the wiki page above.

        Parameters
        ----------
        n : int
            dimension parameter in the formula above. M will be returned as
            an (n^2 , (n * (n + 1))/2) matrix
        """

        M_dup = np.zeros((n**2, n*(n + 1)//2))
        for j in range(1, n+1):
            for i in range(j, n+1):
                u = np.zeros(n*(n+1)//2)
                pos = (j-1) * n + i - j*(j-1)//2 - 1
                u[pos] = 1
                
                T_mat = np.zeros((n, n))
                T_mat[i-1, j-1] = T_mat[j-1, i-1] = 1

                M_dup += np.outer(u, np.ravel(T_mat)).T

        return M_dup
    
    def A_matrix(self,n):
        """
        Returns the A for the L * 1 = 0 (zero  row-sum) and tr(L) = n (trace 
        constraint) are equivalent to A * vech(L) = b.

        Parameters
        ----------
        n : int, number of nodes.
            A will be returned as an (n + 1 , (n * (n + 1))/2) matrix
        """
        A = np.zeros((n+1, n*(n+1)//2))
        for i in range(0, A.shape[0] - 1):
            A[i, :] = self.A_row_i(i, n)
        A[n, 0] = 1
        A[n, np.cumsum(np.arange(n, 1, -1))] = 1

        return A
    
    def b_constraint(self,n):
        """
        Returns the b constraint vector for the constraint A * vech(L) = b, i.e.,
        L * 1 = 0 and tr(L) = n.

        Parameters
        ----------
        n : int, number of nodes.
            b will be returned as a vector of size n+1.
        """
        b = np.zeros(n+1)
        b[n] = n
        return b
    
    def A_row_i(self, i, n):
        """
        Returns the i-th row of the matrix A.

        Parameters
        ----------
        n : int, number of nodes.
            row_i will be returned as a vector of size n*(n+1)//2).
        """
        row_i = np.zeros(n*(n+1)//2)
        if i == 0:
            row_i[np.arange(n)] = 1
        else:
            tmp_vec = np.arange(n-1, n-i-1, -1)
            tmp2_vec = np.append([i], tmp_vec)
            tmp3_vec = np.cumsum(tmp2_vec)
            row_i[tmp3_vec] = 1
            end_pt = tmp3_vec[-1]
            row_i[np.arange(end_pt, end_pt + n-i)] = 1

        return row_i

    def G_matrix(self,n):
        """
        Returns the G matrix vector for the constraint G * vech(L) <= h,
        (h is a zero vector) i.e., L_ij = L_ji <= 0 constraint 
        (non-positive off-diagonal entries).

        Parameters
        ----------
        n : int, number of nodes.
            b will be returned as a vector of size n+1.
        """
        G = np.zeros((n*(n-1)//2, n*(n+1)//2))
        tmp_vec = np.cumsum(np.arange(n, 1, -1))
        tmp2_vec = np.append([0], tmp_vec)
        tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
        for i in range(G.shape[0]):
            G[i, tmp3_vec[i]] = 1

        return G