"""
Least Angle Regression algorithm. See the documentation on the
Generalized Linear Model for a complete discussion.
"""

# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux
#
# License: BSD Style.

import sys

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

from sklearn.linear_model.base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils import array2d, arrayfuncs


def lars_path(X, y, Xy=None, Gram=None, max_iter=500,
              alpha_min=0, method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=False, return_path=True,
              group_ids=None, positive=False):
    """Compute Least Angle Regression and Lasso path

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Parameters
    -----------
    X: array, shape: (n_samples, n_features)
        Input data

    y: array, shape: (n_samples)
        Input targets

    max_iter: integer, optional
        Maximum number of iterations to perform, set to infinity for no limit.

    Gram: None, 'auto', array, shape: (n_features, n_features), optional
        Precomputed Gram matrix (X' * X), if 'auto', the Gram
        matrix is precomputed from the given X, if there are more samples
        than features

    alpha_min: float, optional
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method: {'lar', 'lasso'}
        Specifies the returned model. Select 'lar' for Least Angle
        Regression, 'lasso' for the Lasso.

    eps: float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_X: bool
        If False, X is overwritten.

    copy_Gram: bool
        If False, Gram is overwritten.

    Returns
    --------
    alphas: array, shape: (max_features + 1,)
        Maximum of covariances (in absolute value) at each iteration.

    active: array, shape (max_features,)
        Indices of active variables at the end of the path.

    coefs: array, shape (n_features, max_features + 1)
        Coefficients along the path

    See also
    --------
    lasso_path
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    Notes
    ------
    * http://en.wikipedia.org/wiki/Least-angle_regression

    * http://en.wikipedia.org/wiki/Lasso_(statistics)#LASSO_method
    """
    if group_ids is not None:
        group_ids = np.array(group_ids).copy()
        max_iter = len(np.unique(group_ids))

    n_features = X.shape[1]
    n_samples = y.size
    max_features = min(max_iter, n_features)

    if return_path:
        coefs = np.zeros((max_features + 1, n_features))
        alphas = np.zeros(max_features + 1)
    else:
        coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
        alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    L = np.empty((max_features, max_features), dtype=X.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (X,))

    if Gram is None:
        if copy_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')
    elif Gram == 'auto':
        Gram = None
        if X.shape[0] > X.shape[1]:
            Gram = np.dot(X.T, X)
    elif copy_Gram:
            Gram = Gram.copy()

    if Xy is None:
        Cov = np.dot(X.T, y)
    else:
        Cov = Xy.copy()

    if verbose:
        if verbose > 1:
            print "Step\t\tAdded\t\tDropped\t\tActive set size\t\tC"
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning

    if group_ids is not None:
        selected_group = list()

    while True:
        if Cov.size:
            if group_ids is None:
                if positive:
                    C_idx = np.argmax(Cov)
                else:
                    C_idx = np.argmax(np.abs(Cov))
                C_ = Cov[C_idx]
                if C_ <= 0 and positive:
                    break
                C = np.fabs(C_)
            else:
                if positive:
                    tmp = Cov
                else:
                    tmp = np.abs(Cov)
                already_selected = np.zeros(len(tmp), dtype=np.bool)
                for gid in selected_group:
                    already_selected[group_ids == gid] = True
                tmp[already_selected] = 0.
                C_idx = np.argmax(tmp)
                C_ = Cov[C_idx]
                if C_ <= 0 and positive:
                    break
                C = np.fabs(C_)
                selected_group.append(group_ids[C_idx])
        else:
            C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

        alpha[0] = C / n_samples
        if alpha[0] < alpha_min:  # early stopping
            # interpolation factor 0 <= ss < 1
            if n_iter > 0:
                # In the first iteration, all alphas are zero, the formula
                # below would make ss a NaN
                ss = (prev_alpha[0] - alpha_min) / (prev_alpha[0] - alpha[0])
                coef[:] = prev_coef + ss * (coef - prev_coef)
            alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break

        if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov = Cov[1:]  # remove Cov[0]

            if group_ids is not None:
                group_ids[C_idx], group_ids[0] = group_ids[0], group_ids[C_idx]
                group_ids = group_ids[1:]  # remove group_ids[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            arrayfuncs.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active])
            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print "%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                            n_active, C)
        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                               sign_active[:n_active], lower=True)

        # is this really needed ?
        AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

        if not np.isfinite(AA):
            # L is too ill-conditionned
            i = 0
            L_ = L[:n_active, :n_active].copy()
            while not np.isfinite(AA):
                L_.flat[::n_active + 1] += (2 ** i) * eps
                least_squares, info = solve_cholesky(L_,
                                    sign_active[:n_active], lower=True)
                tmp = max(np.sum(least_squares * sign_active[:n_active]), eps)
                AA = 1. / np.sqrt(tmp)
                i += 1
        least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)


        if group_ids is not None:
            mask = np.ones(group_ids.shape,dtype=bool)
            for g in selected_group:
                mask = mask & (group_ids!=g)

            arg = ((C - Cov) / (AA - corr_eq_dir + tiny))[mask]
            g1 = arrayfuncs.min_pos(arg)
            arg = ((C + Cov) / (AA + corr_eq_dir + tiny))[mask]
            g2 = arrayfuncs.min_pos(arg)
        else:
            g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny))
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny))
        
        if positive: 
            gamma_ = min(g1, C / AA)
        else:
            gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                # resize the coefs and alphas array
                add_features = 2 * max(1, (max_features - n_active))
                coefs.resize((n_iter + add_features, n_features))
                alphas.resize(n_iter + add_features)
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
            alpha = alphas[n_iter, np.newaxis]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
        else:
            # mimic the effect of incrementing n_iter on the array references
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)

        coef[active] = prev_coef[active] + gamma_ * least_squares
        
        
        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            arrayfuncs.cholesky_delete(L[:n_active, :n_active], idx)

            n_active -= 1
            m, n = idx, n_active
            drop_idx = active.pop(idx)

            if group_ids is not None:
                selected_group.remove(group_ids[idx])
                group_ids = np.r_[idx,group_ids]  # remove group_ids[0]
            
            if Gram is None:
                # propagate dropped variable
                for i in range(idx, n_active):
                    X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                    indices[i], indices[i + 1] = \
                            indices[i + 1], indices[i]  # yeah this is stupid

                # TODO: this could be updated
                residual = y - np.dot(X[:, :n_active], coef[active])
                temp = np.dot(X.T[n_active], residual)

                Cov = np.r_[temp, Cov]
            else:
                for i in range(idx, n_active):
                    indices[i], indices[i + 1] = \
                                indices[i + 1], indices[i]
                    Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i + 1])
                    Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                      Gram[:, i + 1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
                residual = y - np.dot(X, coef)
                temp = np.dot(X.T[drop_idx], residual)
                Cov = np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
            if verbose > 1:
                print "%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp))

    if return_path:
        # resize coefs in case of early stop
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]

        return alphas, active, coefs.T
    else:
        return alpha, active, coef
