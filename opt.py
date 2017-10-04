import logging
import sys

import numpy as np
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b

logger = logging.getLogger(__name__)


class OptimizerBase(object):

    def __init__(self, lrate=1.0, 
         n_iter_check=100, n_iter_without_progress=50,
         min_grad_norm=1e-7, min_err_diff=1e-7, verbose=0):
         self._iter = 0
         self.lrate = lrate
         self.n_iter_check = n_iter_check
         self.n_iter_without_progress = n_iter_without_progress
         self.min_grad_norm = min_grad_norm
         self.min_err_diff = min_err_diff
         self.verbose = verbose

    def optimize(self, x0, obj_fun, obj_kwargs={}, num_iters=100, update=None):
        x = x0.copy().ravel()
        self.update = np.zeros_like(x0) if update is None else update
        old_err = np.finfo(np.float).max
        self.best_err = old_err
        self.best_iter = 0

        self._pre_optimize(x)

        for i in range(self._iter, self._iter + num_iters):
            self._iter = i
            new_err, grad = obj_fun(x, **obj_kwargs)
            self.update = self._compute_update(grad)
            x += self.update

            if (i + 1) % self.n_iter_check == 0:
                grad_norm = linalg.norm(grad)
                if self._stop_prematurely(old_err, new_err, grad_norm):
                    break
            old_err = new_err
        return x, new_err

    def _pre_optimize(self, x0):
        pass

    def _compute_update(self, grad):
        raise NotImplementedError

    def _stop_prematurely(self, old_err, new_err, grad_norm):

        if self.verbose >= 2:
            msg = "Iteration %d: err = %.7f, gradient norm = %.7f"
            logger.info(msg % (self._iter + 1, new_err, grad_norm))

        if new_err < self.best_err:
            self.best_err = new_err
            self.best_iter = self._iter
        elif self._iter - self.best_iter > self.n_iter_without_progress:
            if self.verbose >= 2:
                msg = ("Iteration %d: did not make any progress "
                       "during the last %d episodes. Finished.")
                logger.info(msg % (self._iter + 1, self.n_iter_without_progress))
            return True
        if grad_norm <= self.min_grad_norm:
            if self.verbose >= 2:
                msg = "Iteration %d: gradient norm %f. Finished."
                logger.info(msg % (self._iter + 1, grad_norm))
            return True
        err_diff = np.abs(new_err - old_err)
        if err_diff <= self.min_err_diff:
            if self.verbose >= 2:
                msg = "[t-SNE] Iteration %d: error difference %f. Finished."
                logger.info(msg % (self._iter + 1, err_diff))
            return True
        return False


class Adam(OptimizerBase):

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8,
                 *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.beta1_t = 1.0
        self.beta2_t = 1.0

    def _pre_optimize(self, x):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

    def _compute_update(self, grad):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        self.beta1_t = self.beta1_t * self.beta1
        self.beta2_t = self.beta2_t * self.beta2
        m_hat = self.m / (1.0 - self.beta1_t)
        v_hat = self.v / (1.0 - self.beta2_t)

        update = - self.lrate * m_hat / (np.sqrt(v_hat) + self.eps)
        return update


class GD(OptimizerBase):

    def __init__(self, momentum=0.5, *args, **kwargs):
        super(GD, self).__init__(*args, **kwargs)
        self.momentum = momentum

    def _compute_update(self, grad):
        return self.momentum * self.update - self.lrate * grad


class LBFGSB(OptimizerBase):

    def optimize(self, x0, obj_fun, obj_kwargs={}, num_iters=100, update=None):
        # Repackage as float64
        def obj(y):
            f, g = obj_fun(y.astype(np.float32), **obj_kwargs)
            return f, g.astype(np.float64).reshape(-1)

        x, _, dic = fmin_l_bfgs_b(obj, x0, disp=(self.verbose >= 15),
                                  maxiter=num_iters)
        x = x.astype(np.float32)

        self._iter += dic['nit']
        msg = "LBFGS calls made: %d, iters: %d"
        logger.info(msg % (dic['funcalls'], dic['nit']))

        err, grad = obj_fun(x, **obj_kwargs)
        grad_norm = linalg.norm(grad)
        if self.verbose >= 2:
            msg = "Iteration %d: err = %.7f, gradient norm = %.7f"
            logger.info(msg % (self._iter + 1, err, grad_norm))
        return x, err
