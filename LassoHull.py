import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import cd_fast
from sklearn.utils.validation import check_random_state
import cvxopt
import time

class LassoHull(object):
    def __init__(self, nu=1.01, seed=0, max_iter=1000, tol=1e-5, eps=1e-5, eta=-1.0):
        self.nu = nu
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.eta = eta
        self.direc_ = []
        self.Btime_ = []
        self.B_ = []
        self.Bobj_ = []
        self.dist_ = []
        self.C_ = None
        self.Ctime_ = []
        self.Ccount_ = []
    
    def set_data(self, X, y, rho):
        self.X = X
        self.y = y
        self.rho = rho
        self.beta_ = self.__fit_lasso(X, y, rho, max_iter=self.max_iter, tol=self.tol)
        self.obj_ = self.__lasso_obj(self.beta_, X, y, rho)
        if self.eta >= 0:
            C = np.linalg.inv(X.T.dot(X) / X.shape[0] + self.eta * np.identity(X.shape[1]))
            self.Q_ = np.linalg.cholesky(C).T
        else:
            self.Q_ = np.identity(X.shape[1])
    
    
    def add_extreme(self, M=1, verbose=-1):
        p = self.X.shape[1]
        for m in range(M):
            if verbose > 0 and np.mod(m, verbose)==0:
                print(m)
            np.random.seed(self.seed)
            d = np.random.randn(1, p).dot(self.Q_)[0, :]
            d /= np.linalg.norm(d)
            self.direc_.append(d)
            start = time.time()
            beta = self.__find_lasso_extreme(self.X, self.y, self.rho, d, self.nu * self.obj_, max_iter=self.max_iter, tol=self.tol, eps=self.eps)
            elapsed_time = time.time() - start
            self.B_.append(beta)
            self.Btime_.append(elapsed_time)
            self.Bobj_.append(self.__lasso_obj(beta, self.X, self.y, self.rho))
            self.seed += 1
    
    
    def initialize_hull(self, seed=None):
        self.idx_ = [j for j in range(len(self.B_))]
        if seed is None:
            i = np.argmax([np.linalg.norm(self.beta_ - b) for b in self.B_])
        else:
            np.random.seed(seed)
            i = np.random.randint(len(self.B_))
        self.C_ = self.B_[i][:, np.newaxis]
        self.dist_ = []
        self.idx_.remove(i)
        self.__dist_ub = np.array([np.infty] * (len(self.idx_) + 1))
        self.__dist_ub[i] = 0
    
    
    def add_vertex(self, K=10):
        for k in range(K):
            if len(self.idx_) == 0:
                break
            i_max = 0
            d_max = 0
            count = 0
            start = time.time()
            jdx = np.argsort(self.__dist_ub[self.idx_])[::-1]
            for i in jdx:
                ii = self.idx_[i]
                if d_max > self.__dist_ub[ii]:
                    break
                count += 1
                b = self.project_to_hull(self.B_[ii])
                d = np.linalg.norm(self.B_[ii] - b)
                self.__dist_ub[ii] = d
                if d > d_max:
                    i_max = ii
                    d_max = d
            elapsed_time = time.time() - start
            self.C_ = np.c_[self.C_, self.B_[i_max]]
            self.__dist_ub[i_max] = 0
            self.Ctime_.append(elapsed_time)
            self.Ccount_.append(count)
            self.dist_.append(d_max)
            self.idx_.remove(i_max)
    
    
    def project_to_hull(self, beta):
        B = self.C_
        m = B.shape[1]
        P = cvxopt.matrix(B.T.dot(B))
        q = cvxopt.matrix(-B.T.dot(beta))
        A = cvxopt.matrix(np.ones((1,m)))
        b = cvxopt.matrix(np.array([1.0]))
        G = cvxopt.matrix(-np.identity(m))
        h = cvxopt.matrix(np.zeros(m))
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P,q,A=A,b=b,G=G,h=h)
        ans = np.transpose(np.array(sol['x'])).dot(B.T)
        return ans[0]
    
    
    def __lasso_obj(self, beta, X, y, rho):
        n = y.size
        res = y - X.dot(beta)
        return 0.5 * res.dot(res) / n + rho * np.linalg.norm(beta, 1)
    
    
    def __fit_lasso(self, X, y, rho, max_iter=1000, tol=1e-5):
        lasso = Lasso(fit_intercept=False, alpha=rho, max_iter=max_iter, tol=tol)
        lasso.fit(X, y)
        return lasso.coef_
    
    def __fit_lasso_withlin(self, XX, Xy, yn, rho, d, lam, max_iter=1000, tol=1e-5):
    
        # vectors & matrices
        n = yn.size
        p = Xy.size
        Q = XX * lam / n
        q = Xy * lam / n + d

        # fit
        rng = check_random_state(0)
        beta = np.zeros(p)
        res = cd_fast.enet_coordinate_descent_gram(beta, lam * rho, 0, Q, q, yn, max_iter, tol, rng)
        return res[0]
    
    
    def __find_lasso_extreme(self, X, y, rho, d, nu, max_iter=1000, tol=1e-5, eps=1e-5):

        # vectors & matrices
        XX = X.T.dot(X)
        Xy = X.T.dot(y)
        yn = y / np.sqrt(y.size)

        # binary search
        lam_prev = 1.0
        beta = self.__fit_lasso_withlin(XX, Xy, yn, rho, d, lam_prev, max_iter=max_iter, tol=tol)
        obj_prev = self.__lasso_obj(beta, X, y, rho)
        if obj_prev >= nu:
            lam = lam_prev * 2.0
        else:
            lam = lam_prev * 0.5
        while True:
            beta = self.__fit_lasso_withlin(XX, Xy, yn, rho, d, lam, max_iter=max_iter, tol=tol)
            obj = self.__lasso_obj(beta, X, y, rho)
            if obj >= nu:
                lam_prev = lam
                lam *= 2.0
            else:
                lam_prev = lam
                lam *= 0.5
            if (obj >= nu and obj_prev < nu) or (obj_prev >= nu and obj < nu):
                break
        lam_lb = min(lam, lam_prev)
        lam_ub = max(lam, lam_prev)
        while True:
            lam = 0.5 * (lam_lb + lam_ub)
            beta = self.__fit_lasso_withlin(XX, Xy, yn, rho, d, lam, max_iter=max_iter, tol=tol)
            obj = self.__lasso_obj(beta, X, y, rho)
            if obj >= nu:
                lam_lb = lam
            else:
                lam_ub = lam
            if obj <= nu and nu - obj < eps:
                break
        return beta