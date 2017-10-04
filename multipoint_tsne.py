import datetime as dt
import logging
import sys
from collections import defaultdict

import numpy as np
import theano
import theano.tensor as tt
from scipy.spatial.distance import squareform
from sklearn import utils
from sklearn.decomposition import PCA
from sklearn.manifold import t_sne#, _barnes_hut_tsne
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree

import opt
import pyximport
pyximport.install(reload_support=True)
import _barnes_hut_tsne
import _barnes_hut_mptsne

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%H:%M",
                    level=logging.INFO, stream=sys.stdout)


def pdist2(X, Y=None, lib=np):
    """Computes a matrix of squared distances.

    Args:
        X: Matrix, each row is a datapoint.
        Y: (optional) Matrix of datapoints.
        lib: (optional) Module computing the expression (numpy or Theano)
    """
    sum_X = lib.sum(lib.square(X), 1)
    if Y is None:
        return (-2 * lib.dot(X, X.T) + sum_X).T + sum_X
    sum_Y = lib.sum(lib.square(Y), 1)
    return (-2 * lib.dot(Y, X.T) + sum_X).T + sum_Y

def pdist2_with_copies(n_in, Y, full_copy_mask):
    """Computes a matrix of squared distances.

    Points in Y might have copies. If y_i and y_j have copies, then the computed
    distance should be the closest distance of any of their copies.

    If Y has N+C points, where C is the number of copies, returned matrix
    should be N x N.

    Args:
        Y: Matrix of datapoints.
        full_copy_mask: List of N+C indices describing which datapoint has been
            copied.
    """
    if full_copy_mask is None:
        return pdist2(Y)

    count_dict = defaultdict(int)
    layer_specs = defaultdict(list)
    for ind,proto in enumerate(full_copy_mask):
        count_dict[proto] += 1
        l = count_dict[proto]
        layer_specs[l].append((ind, proto))

    # Sort the last layer. Each preceeding layer should have elements
    # from the previous layer in the same order at the end,
    # the rest at the beginning, e.g.:
    #     layer#1: [0, 1, 4, 3, 5, 2]
    #     layer#2: [4, 3, 5, 2]
    #     layer#3: [3, 5, 2]
    #     layer#4: [2]
    nlayers = len(layer_specs.keys())
    for l in range(1, nlayers + 1):
        layer_specs[l] = sorted(
            layer_specs[l],
            key=lambda (ind, proto): count_dict[proto] * 1000000 + proto)

    # Compute pdist between original pts
    inds = list(zip(*layer_specs[1])[0])
    assert len(inds) == n_in

    Layer = Y[inds]
    YY = pdist2(Layer, lib=tt)

    # Update it with pdists from copies using tt.minimum
    for i in xrange(1, nlayers + 1):
        for j in xrange(i, nlayers + 1):

            if i == j == 1:
                continue

            layer1_inds = list(zip(*layer_specs[i])[0])
            layer2_inds = list(zip(*layer_specs[j])[0])

            Layer1 = Y[layer1_inds]
            Layer2 = Y[layer2_inds]
            sz_layer1 = Layer1.shape[0]
            sz_layer2 = Layer2.shape[0]
            YK = pdist2(Layer1, Layer2, lib=tt)

            YY = tt.set_subtensor(YY[-sz_layer1:, -sz_layer2:],
                                  tt.minimum(YY[-sz_layer1:, -sz_layer2:], YK))
            if i != j:
                YY = tt.set_subtensor(YY[-sz_layer2:, -sz_layer1:],
                                      tt.minimum(YY[-sz_layer2:, -sz_layer1:], YK.T))
    pdist_Y = YY

    # Shuffle it back, so the order would match P.
    inv_perm = np.argsort(inds)
    pdist_Y = pdist_Y[inv_perm,:][:,inv_perm]
    return pdist_Y

def Q_graph(pdist2):
    Q = (1.0 / (1.0 + pdist2))
    Q = tt.set_subtensor(
        Q[tt.arange(Q.shape[0]), tt.arange(Q.shape[0])], 0.0)
    Q /= Q.sum()
    Q = tt.maximum(Q, 1e-12)
    return Q

def full_copy_mask__to__next_repr_pos_idx(fcm, N):
    ret = np.ones(fcm.shape, dtype=np.int64) * -1
    num_reprs = np.ones(N, dtype=np.int32)
    next_idx_for_label = np.arange(N)
    for idx in range(N, fcm.shape[0]):
        label = fcm[idx]
        num_reprs[label] += 1
        ret[next_idx_for_label[label]] = idx
        next_idx_for_label[label] = idx
    return ret, num_reprs

def _prepare_data(x, p, perplexity, initial_dims, compute_neighbors_nn=False):
    neighbors_nn = None

    if x is None and p is None:
        raise ValueError('Both supplied X and P are None.')
    if not x is None and not p is None:
        raise ValueError('Supplied both X and P. Please supply one.')
    if p is None:
        logger.info('Applying PCA')
        time0 = dt.datetime.now()
        if x.shape[1] > initial_dims:
            x = PCA(n_components=initial_dims).fit_transform(x)
        time1 = dt.datetime.now()
        if (time1 - time0).seconds > 10:
            logging.info('Took', (time1 - time0).seconds, 'seconds')

        logging.info('Computing pairwise dists...')
        time0 = dt.datetime.now()
        distances = pairwise_distances(x, metric='euclidean', squared=True)
        n_samples = x.shape[0]
        k = min(n_samples - 1, int(3. * perplexity + 1))
        time1 = dt.datetime.now()
        if (time1 - time0).seconds > 10:
            logging.info('Took', (time1 - time0).seconds, 'seconds')
        
        logging.info('Computing NNs...')
        time0 = dt.datetime.now()
        bt = BallTree(x)
        distances_nn, neighbors_nn = bt.query(x, k=k + 1)
        neighbors_nn = neighbors_nn[:, 1:]
        time1 = dt.datetime.now()
        if (time1 - time0).seconds > 10:
            logging.info('Took', (time1 - time0).seconds, 'seconds')
        
        logging.info('Computing P...')
        time0 = dt.datetime.now()
        P_sparse = tsne._joint_probabilities_nn(distances, neighbors_nn,
                                                perplexity, verbose=False)
        time1 = dt.datetime.now()
        if (time1 - time0).seconds > 10:
            logging.info('Took', (time1 - time0).seconds, 'seconds')
        
        logging.info('Computing squareform P...')
        time0 = dt.datetime.now()
        p = squareform(P_sparse).astype(np.float32)
        time1 = dt.datetime.now()
        if (time1 - time0).seconds > 10:
            logger.info('Took', (time1 - time0).seconds, 'seconds')

    if compute_neighbors_nn and neighbors_nn is None:
        n_in = x.shape[0] if p is None else p.shape[0]
        neighbors_nn = k_neighbors(perplexity, n_in, x=x, p=p)

    return np.ascontiguousarray(p), np.ascontiguousarray(neighbors_nn)

def copy_potential(p, q, y, copy_mask, sne_grad=True):
    pdist = pdist2(y)

    n_in = p.shape[0]
    n_out = y.shape[0]

    # Iterate over nonnegative values and check,
    # if the corresponding pts are the closest (copies).
    if not copy_mask is None:
        cm = copy_mask # CopyMask(n_in)
        # cm.update(copy_mask)
        p_ = p[cm.full_copy_mask, :][:, cm.full_copy_mask]
        q_ = q[cm.full_copy_mask, :][:, cm.full_copy_mask]
        const_terms = (p_ > q_) * (p_ - q_)
        if not sne_grad:
            const_terms *= (1.0 / (1.0 + pdist))
        for i in xrange(n_out):
            for j in xrange(n_out):
                if i == j or const_terms[i,j] < 1e-14:
                    continue
                # Check if pts for (i,j) are the closest ones.
                if not cm.are_closest(y, i, j):
                    const_terms[i,j] = 0.0
    else:
        const_terms = (p > q) * (p - q)
        if not sne_grad:
            const_terms *= (1.0 / (1.0 + pdist))

    # Now only Fattr for valid copy-copy pairs are left in const terms.
    # All that's left is to multiply by y_i - y_j parts and sum up.
    copy_potential_grads = np.zeros(y.shape)
    for i in xrange(n_out):
        g = const_terms[i][:, None] * (y[i] - y)
        copy_potential_grads[i] = np.sum(g, axis=0)
        # Now g holds all Fattr forces working on point i.
    return -1.0 * copy_potential_grads

# Either use cython or pass neighbors; check it inside the function
def cleanup_low_pbb_mass(targets, p, y, cm, neighbors=None, mass_threshold=0.05):
    before = len(cm._copy_mask)
    if before == 0:
        return targets

    assert not neighbors is None, 'Neighbors cannot be None'
    next_repr_pos_idx, num_reprs = full_copy_mask__to__next_repr_pos_idx(
        cm.full_copy_mask, p.shape[0])
    mass = np.zeros(y.shape[0], dtype=np.float32)
    _barnes_hut_mptsne.compute_pbb_mass(p, y, neighbors, num_reprs, 
                                        next_repr_pos_idx, mass, verbose=1)
    inds2del = []
    for r in cm.copy_group_iter():
        if len(r) == 1:
            continue
        m = np.asarray(map(lambda t: mass[t], r))
        m /= m.sum()
        inds = np.asarray(r)[m < mass_threshold]
        # Note: Inds should be deleted starting from the highest!
        # Deletion of protos is carried by swapping with one of copies.
        # It may mess up, if a proto would be deleted first.
        for i in np.sort(inds)[::-1]:
            inds2del.append(i)
    for i in sorted(inds2del)[::-1]:
        targets = del_pt(i, p, cm, targets)
    after = len(cm._copy_mask)
    logger.info('Cleanup of copies: {} -> {}'.format(before, after))
    return targets

def probability_mass(p, y, cm):
    n = p.shape[0]
    mass = np.zeros((y.shape[0]))
    pd = pdist2(y)

    for i in xrange(y.shape[0]):
        pts_i = cm.copies_of(i)
        proto_i = i if i < n else cm._copy_mask[i - n]
        for j in xrange(y.shape[0]):
            proto_j = j if j < n else cm._copy_mask[j - n]
            if p[proto_i, proto_j] < 1e-8:
                continue
            pts_j = cm.copies_of(j)

            # Determine if d(i,j) is the closest one for this pair
            min_dist = np.min(pd[pts_i, :][:, pts_j])
            if np.isclose(pd[i][j], min_dist):
                mass[i] += p[proto_i][proto_j]
    return mass

def del_pt(ind, p, cm, targets=[]):
    """Attempt to delete a copy of a point (or proto and replace with copy)."""
    n = p.shape[0]
    if ind < n:
        group = cm.copies_of(ind)
        if len(group) == 1:
            raise ValueError('Point does not have copies to delete from')
        assert(ind == group[0])
        for y in targets:
            swap_pt(y, ind, group[1])
        ind = group[1]
    # Now point 'ind' can be safely deleted.
    assert(ind >= n)
    cm.overwrite(np.delete(cm._copy_mask, [ind - n], axis=0))
    return [np.delete(y, [ind], axis=0) for y in targets]

def swap_pt(y, ind1, ind2):
    tmp_row = np.copy(y[ind1])
    y[ind1] = np.copy(y[ind2])
    y[ind2] = np.copy(tmp_row)

def k_neighbors(perplexity, n_in, x=None, p=None, verbose=True):
    k = min(n_in - 1, int(3. * perplexity + 1))
    neighbors_nn = None
    if verbose:
        logging.info("Computing %i nearest neighbors..." % k)
    if x is None:
        # Use the precomputed distances to find
        # the k nearest neighbors and their distances
        neighbors_nn = np.argsort(p, axis=1)[:,::-1][:, :k]
    else:
        raise NotImplementedError
        # Find the nearest neighbors for every point
        bt = BallTree(X)
        # LvdM uses 3 * perplexity as the number of neighbors
        # And we add one to not count the data point itself
        # In the event that we have very small # of points
        # set the neighbors to n - 1
        distances_nn, neighbors_nn = bt.query(X, k=k + 1)
        neighbors_nn = neighbors_nn[:, 1:]
    return np.ascontiguousarray(neighbors_nn)


class CopyMask(object):
    """Represents a mapping of datapoints to their protos (progenitors).

    Args:
        n: int, initial number of datapoints (without any copies)
    """
    def __init__(self, n):
        self.n = n
        self._copy_mask = np.ndarray((0,)).astype(np.int32)
        self.proto_to_copies = defaultdict(list)
        self.layer_specs = defaultdict(list)
        self.update(self._copy_mask)

    @property
    def full_copy_mask(self):
        return np.concatenate([np.arange(self.n), self._copy_mask])

    @property
    def num_copies(self):
        return self._copy_mask.shape[0]

    def max_ncopies(self):
        if len(self._copy_mask) == 0:
            return 1
        return np.max(np.bincount(self._copy_mask)) + 1

    def update(self, new_copy_mask):
        self._copy_mask = np.hstack([self._copy_mask, new_copy_mask])
        self._copy_mask = self.proto_only(self._copy_mask)
        self.proto_to_copies = defaultdict(list)
        self.layer_specs = defaultdict(list)
        for ind,proto in enumerate(self.full_copy_mask):
            self.proto_to_copies[proto].append(ind)
            l = len(self.proto_to_copies[proto])
            self.layer_specs[l].append((ind,proto))

    def overwrite(self, new_copy_mask):
        self._copy_mask = np.ndarray((0,)).astype(np.int32)
        self.update(new_copy_mask)

    def proto_only(self, new_copy_mask):
        ret = np.copy(new_copy_mask)
        for i,v in enumerate(ret):
            ret[i] = self.full_copy_mask[v]
        return ret

    def copies_of(self, k):
        """All copies of pt with index k"""
        proto = self.full_copy_mask[k]
        ret = self.proto_to_copies[proto]
        assert (k in ret)
        return ret

    def copy_group_iter(self):
        for i in xrange(self.n):
            yield self.copies_of(i)

    def copy_pair_iter(self):

        copy_dict = defaultdict(list)
        # Set pbbs among copies.
        for ind,proto in enumerate(self._copy_mask):
            copy_dict[proto].append(ind + self.n)

        for proto,copy_list in copy_dict.items():
            copy_list.append(proto)
            for c1 in copy_list:
                for c2 in copy_list:
                    yield (c1, c2)

    def are_closest(self, y, i, j):
        """Checks if datapoints y_i and y_j are the closest ones
           of their corresponding copy groups.
        """
        ci = self.copies_of(i)
        cj = self.copies_of(j)
        if len(ci) == len(cj) == 1:
            return True
        if i in cj:
            return False

        pdist = pdist2(y[ci], y[cj])
        min_pt = np.unravel_index(np.argmin(pdist), (len(ci), len(cj)))
        return ci[min_pt[0]] == i and cj[min_pt[1]] == j


class MultipointTSNE(object):

    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=4.0,
                 learning_rate=200.0,
                 optimizer='gd',
                 optimizer_kwargs={},
                 initial_dims=50, 
                 train_schedule=[(True, 250, 0.0), (False, 750, 0.0)],
                 init='random',
                 verbose=0,
                 n_iter_check=100, 
                 random_state=None,
                 method='barnes_hut',
                 angle=0.5,
                 num_cleanups=2,
                 cleanup_thresh=0.15, 
                 cleanup_thresh_after_clonning=None):

        self.ndims = n_components
        self.perplexity = perplexity
        self.early_exagg = early_exaggeration
        optimizer_kwargs.update({'lrate': learning_rate, 'verbose': verbose,
                                 'n_iter_check': n_iter_check})
        opt_protos = {'gd': opt.GD, 'adam': opt.Adam, 'lbfgsb': opt.LBFGSB}
        self.optimizer = opt_protos[optimizer](**optimizer_kwargs)
        self.initial_dims = initial_dims
        self.train_schedule = train_schedule
        self.init_method = init
        self.verbose = verbose
        self.random_state = utils.check_random_state(random_state)
        self.method = method
        if method != 'barnes_hut':
            raise NotImplementedError
        self.angle = angle

        self.cleanup_thresh = cleanup_thresh
        if cleanup_thresh_after_clonning is None:
            cleanup_thresh_after_clonning = cleanup_thresh
        self.cleanup_thresh_after_clonning = cleanup_thresh_after_clonning

        self.num_cleanups = num_cleanups

        self.history = []
        self.copy_history = [] # Iterations, in which points were clonned.

    def _initial_solution(self, n, method, x=None):
        if method == 'random':
            return self.random_state.randn(n, self.ndims)
        elif method == 'svd':
            from sklearn.decomposition import TruncatedSVD
            return np.TruncatedSVD(n_components=self.ndims).fit_transform(x)
        else:
            raise ValueError('Unknown initialization method ' + method)

    def compute_Q(self, y=None, copy_mask=None):
        if y is None:
            y = self.y

        if copy_mask is None and self.cm is None:
            pdist2_y = pdist2(y)
        else:
            if copy_mask is None:
                copy_mask = self.cm._copy_mask
            n = y.shape[0] - len(copy_mask)
            pdist2_y = pdist2_with_copies(
                self.n_in, y, self.cm.full_copy_mask)

        Pdist = tt.dmatrix('Pdist')
        return Q_graph(Pdist).eval({Pdist: pdist2_y})

    def copy_pts(self, copy_perc=None, ncopies=None, copy_potential_grads=None,
                 method='percent'):
        """Sums forces coming from points, for which Fattr > Frep.

         j-th point is similar in original space (P[i,j] > 0 so Fattr exists).
         The point does not meet similarity yet (attr stornger than rep).
        """
        self.copy_history.append(len(self.history))
        logger.info('Picking pts to copy')

        if copy_potential_grads is None: 
            if self.method == 'barnes_hut':
                next_repr_pos_idx, num_reprs = full_copy_mask__to__next_repr_pos_idx(
                    self.cm.full_copy_mask, self._P.shape[0]
                )
                grad = np.zeros(self.y.shape, dtype=np.float32)
                copy_potential_grads = np.zeros(self.y.shape, dtype=np.float32)
                _barnes_hut_mptsne.gradient_mptsne(
                    self._P, self.y, self.neighbors_nn, grad, copy_potential_grads, 
                    num_reprs, next_repr_pos_idx, self.angle, self.ndims, False,
                    correct_cell_counts=True, dof=1.0, skip_num_points=0
                )
            elif method == 'exact':
                q = self.compute_Q(y=self.y)
                copy_potential_grads = copy_potential(self._P, q, self.y, self.cm)
            else:
                raise ValueError
            self.copy_potential_grads = copy_potential_grads

        if method == 'auto':
            potential = np.sum(copy_potential_grads ** 2, axis=1)
            t = 0.0005
            ncopies = np.sum(potential > t)
        elif method == 'percent':
            # Select top copy_percentage.
            ncopies = int(round(copy_perc * self._P.shape[0] / 100))
        else:
            raise ValueError('Unknown copy method ' + method)

        init_ncopies = ncopies
        if ncopies > 0:
            # Reverse and select first ncopies.
            potential = np.sum(self.copy_potential_grads ** 2, axis=1)
            cm_update = np.argsort(potential)[::-1][:ncopies]
    
            if method == 'percent':
                method += '%d ' % copy_perc
            (y_update, cm_update) = self.initialize_copies(self.y, cm_update)
            self.y = np.vstack([self.y, y_update])
            self.cm.update(cm_update)
            ncopies = len(cm_update)

            def update_param_(param_, with_avg=False):
                if with_avg:
                    new_rows = np.tile(
                        np.mean(param_.reshape(-1, self.ndims), axis=0),
                        (ncopies, 1))
                else:
                    np.zeros((ncopies, self.ndims), dtype=param_.dtype)
                return np.vstack([
                    param_.reshape(-1, self.ndims), 
                    new_rows,
                ]).ravel()
            if hasattr(self.optimizer, 'momentum'):
                self.optimizer.update = update_param_(self.optimizer.update)
            if hasattr(self.optimizer, 'm'):
                self.optimizer.m = update_param_(self.optimizer.m, with_avg=True)
                self.optimizer.v = update_param_(self.optimizer.v, with_avg=True)

        logger.info('Initialized %d new copies (%d -> %d -> %d total, %s)' % (
            ncopies, self.y.shape[0] - ncopies, 
            self.y.shape[0] - ncopies + init_ncopies,
            self.y.shape[0], method))

    def initialize_copies(self, y, cm_update):
        p = self._P
        n = p.shape[0]
        newcopies = len(cm_update)
        old_copy_mask = self.cm._copy_mask
        cm_update_proto = self.cm.proto_only(cm_update)
        tmp_fcm = np.concatenate([self.cm.full_copy_mask, cm_update_proto])

        # Cost function for a single point
        Y = tt.dmatrix('Y')
        pdist_Y = pdist2_with_copies(self.n_in, Y, tmp_fcm)
        Q = Q_graph(pdist_Y)

        # Compute for selected points and do not sum up
        KL = tt.where(abs(p) > 1e-8, p * tt.log(p / Q), 0)
        cost_fun = theano.function([Y], [KL.sum(axis=1)])

        y_new = np.copy(y[cm_update])
        yy = np.vstack([y, y_new])

        # Pick appriopriate gradients
        g = self.copy_potential_grads[cm_update]
        assert(g.shape[0] == newcopies)

        # Normalize the gradients
        g /= np.sqrt(np.sum(g ** 2, axis=1))[:,None]

        std = np.std(y, axis=0)
        nsteps = 100
        step_size = 5.0 * std / nsteps 
        costs = np.zeros((nsteps, newcopies))
        yy[-newcopies:] += g * step_size
        for i in range(nsteps):
            yy[-newcopies:] += g * step_size
            # Pick the ones matching this protos
            costs[i] = cost_fun(yy)[0][cm_update_proto]
        steps = np.argmin(costs, axis=0)
        y_new += g * (steps[:, None]+1) * step_size
        return (y_new, cm_update_proto)

    def continue_run(self, new_train_schedule):
        self.run_schedule(new_train_schedule)
        self.train_schedule += new_train_schedule

    def run_schedule(self, schedule):
        niters = sum([stage[1] for stage in schedule])
        logger.info('Running %d iterations' % niters)

        def cleanup(threshold=None):
            if threshold is None:
                threshold = self.cleanup_thresh
            if not threshold or len(self.cm.full_copy_mask) <= self.n_in:
                return
            for j in range(self.num_cleanups):
                targets = [self.y]
                if hasattr(self.optimizer, 'momentum'):
                    targets.append(self.optimizer.update.reshape(-1, self.ndims))
                if hasattr(self.optimizer, 'm'):
                    targets.append(self.optimizer.m.reshape(-1, self.ndims))
                    targets.append(self.optimizer.v.reshape(-1, self.ndims))
                targets = cleanup_low_pbb_mass(
                    targets, self._P, self.y, self.cm,
                    neighbors=self.neighbors_nn, mass_threshold=threshold
                )
                self.y = targets.pop(0)
                if hasattr(self.optimizer, 'momentum'):
                    self.optimizer.update = targets.pop(0).ravel()
                if hasattr(self.optimizer, 'm'):
                    self.optimizer.m = targets.pop(0).ravel()
                    self.optimizer.v = targets.pop(0).ravel()

        for (exaggerate, num_iters, copy_perc) in schedule:
            if copy_perc:
                method = 'auto' if self.autocopy and not exaggerate else 'percent'
                self.copy_pts(copy_perc=copy_perc, method=method)
            cleanup(self.cleanup_thresh_after_clonning)

            if exaggerate:
                self._P *= self.early_exagg
            self.y, err = self.optimizer.optimize(
                self.y, obj_fun=self._kl_divergence_bh,
                obj_kwargs=self.obj_kwargs, num_iters=num_iters)
            self.y = self.y.reshape(-1, self.ndims)

            print_precise_error = 0
            if print_precise_error:
                # Print Theano error
                P, Y = tt.fmatrices('P', 'Y')
                obj = self.tt_cost_copy(self._P, Y, self.cm)
                logging.info('Precise error:', obj(self.y)[0])
            if exaggerate:
                self._P /= self.early_exagg
            cleanup()

    def run(self, x=None, p=None):
        p, self.neighbors_nn = _prepare_data(
            x, p, self.perplexity, self.initial_dims, compute_neighbors_nn=True)
        self._P = np.copy(p).astype(np.float32)
        self.n_in = p.shape[0]
        self.cm = CopyMask(self.n_in)
        self.obj_kwargs = {'P': self._P,
                           'neighbors': self.neighbors_nn,
                           'degrees_of_freedom': 1.0, # XXX
                           'n_components': self.ndims,
                           'angle': self.angle,
                           'skip_num_points': 0, # XXX
                           'verbose': self.verbose,
                           'copy_mask': self.cm}
        self.autocopy = False
        self.y = self._initial_solution(
            self.n_in, self.init_method).astype(np.float32)
        logger.info('Optimizer: %s' % type(self.optimizer))
        self.run_schedule(self.train_schedule)
        return self.y

    def initialize_copies_seq(self, y, cm_update):
        '''Initialize copies sequentially

           New copy_mask has indices of points to be initialized.
           They are in order of importance; try doing line search with
           each one of them, going from the first to the last.
        '''
        p = self._P
        n_in = p.shape[0]
        n_out_prev = y.shape[0]
        cm_update_proto = self.cm.proto_only(cm_update)

        # Precompute stuff about y
        std = np.std(y, axis=0)
        nsteps = 50
        step_size = 5.0 * std / nsteps 
        costs = np.zeros(nsteps)

        yy = y
        # For each pt in copy_mask:
        sys.stdout.write('[c-SNE] Initializing copies ') ; sys.stdout.flush()
        tmp_fcm = np.copy(self.cm.full_copy_mask)
        for ind, (copy, proto) in enumerate(zip(cm_update, cm_update_proto)):
            # Add point to y
            y_new = np.copy(y[copy])
            yy = np.vstack([yy, y_new])
            # Perform line search
            tmp_fcm = np.append(tmp_fcm, proto)
            next_repr_pos_idx, num_reprs = \
                full_copy_mask__to__next_repr_pos_idx(tmp_fcm, n_in)

            # Pick appriopriate gradients
            g = self.copy_potential_grads[copy]
            # Normalize the gradients
            g /= np.sqrt(np.sum(g ** 2))
            for i in range(nsteps):
                yy[-1] += g * step_size
                costs[i] = _barnes_hut_mptsne.estimate_error(
                    p, yy, self.neighbors_nn, num_reprs, next_repr_pos_idx,
                    self.sum_Q, self.ndims, verbose=0)

            step = np.argmin(costs, axis=0)
            if 1 < step + 1 < nsteps:
                sys.stdout.write('.') ; sys.stdout.flush()
                yy[-1] = y[copy] + g * (step + 1) * step_size
            else:
                sys.stdout.write('x') ; sys.stdout.flush()
                yy = yy[:-1]
                tmp_fcm = tmp_fcm[:-1]
        print ''
        return (yy[n_out_prev:], tmp_fcm[n_out_prev:])

    def _kl_divergence_bh(self, Y, P, neighbors, degrees_of_freedom,
                          n_components, angle=0.5, skip_num_points=0,
                          verbose=False, copy_mask=None):
        assert np.all(np.isfinite(Y)), 'Infinite elems in Y'
        if np.any(np.invert(np.isfinite(Y))):
            print np.where(np.invert(np.isfinite(Y)))
    
        Y = Y.reshape(Y.shape[0] / n_components, n_components)
   
        if copy_mask is None or copy_mask.num_copies == 0:
            grad = np.zeros(Y.shape, dtype=np.float32)
            error = _barnes_hut_tsne.gradient(
                P, Y, neighbors, grad, angle, n_components, verbose=False, 
                dof=degrees_of_freedom, skip_num_points=skip_num_points)
        else:
            # Prepare the 'num_reprs' list
            # Preapre the 'next_repr_pos_id' list
            next_repr_pos_idx, num_reprs = full_copy_mask__to__next_repr_pos_idx(
                copy_mask.full_copy_mask, P.shape[0])
    
            grad = np.zeros(Y.shape, dtype=np.float32)
            potential_grads = np.zeros(Y.shape, dtype=np.float32)
            sum_Q = np.zeros(1, dtype=np.float32)

            error = _barnes_hut_mptsne.gradient_mptsne(
                P, Y, neighbors, grad, potential_grads, num_reprs,
                next_repr_pos_idx, angle, n_components, verbose=False,
                correct_cell_counts=True, dof=degrees_of_freedom,
                skip_num_points=skip_num_points, store_sum_Q=sum_Q)

            self.potential_grads = potential_grads
            self.sum_Q = sum_Q[0]
    
        c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
        grad = grad.ravel()
        grad *= c
    
        return error, grad
