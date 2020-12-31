import ctypes
from math import gamma, erfc
import numba as nb
from numba import njit, vectorize, uint64
from numba.extending import get_cython_function_address
import numpy as np


addr = get_cython_function_address("scipy.special.cython_special", "chdtrc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
chdtrc_fn = functype(addr)


@vectorize('float64(float64, float64)')
def vec_chdtrc(x, y):
    return chdtrc_fn(x, y)

@njit
def chdtrc(x, y):
    return vec_chdtrc(x, y)

###############################
##### Independence Tests ######
###############################


@njit
def kruskal(x, y):
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata, use_missing=False)
    ties = tiecorrect(ranked[ranked>1])
    if ties == 0:
        raise ValueError('All numbers are identical in kruskal')
    # Split the ranks between x and y
    rank_x = ranked[0: len(x)]
    rank_y = ranked[len(x): len(x)+len(y)]
    # Missing values are ranked at -1. Filter them out
    rank_x = rank_x[rank_x>0]
    rank_y = rank_y[rank_y>0]
    # Compute sum^2/n for each group and sum
    ssbn = np.sum(rank_x)**2 / len(rank_x) + np.sum(rank_y)**2 / len(rank_y)
    totaln = len(rank_x) + len(rank_y)
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    df = 1
    h /= ties
    return h, chdtrc(df, h)


@njit
def chi_test(var1, var2, correction=True):
    """ Computes a chi-squared test for two variables. """
    assert var1.size == var2.size, "Both variables should have the same number of observations."
    obs = ~np.isnan(var1*var2)  # Exclude missing values and convert to int
    var1, var2 = np.asarray(var1[obs], dtype=np.int64).copy(), np.asarray(var2[obs], dtype=np.int64).copy()
    cat1, _ = unique(var1)
    cat2, _ = unique(var2)
    observed = np.zeros((cat1.size, cat2.size), dtype=np.float64)
    for i, c1 in enumerate(cat1):
        for j, c2 in enumerate(cat2):
            observed[i, j] = np.sum((var1==c1) * (var2==c2))

    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    expected = expected_freq(observed)
    if np.any(expected == 0):
        raise ValueError("The internally computed table of expected "
                         "frequencies has a zero element.")

    # The degrees of freedom
    dof = expected.size - expected.shape[0] - expected.shape[1] + expected.ndim - 1

    if dof == 0:
        # Degenerate case; this occurs when `observed` is 1D (or, more
        # generally, when it has only one nontrivial dimension).  In this
        # case, we also have observed == expected, so chi2 is 0.
        chi2 = 0.0
        p = 1.0
    else:
        if dof == 1 and correction:
            # Adjust `observed` according to Yates' correction for continuity.
            sign = np.sign(expected - observed)/2
            observed = observed + sign

    h = ((observed - expected)**2/expected).sum()
    p = chdtrc(dof, h)

    return p


@njit
def kendalltau(x, y, use_ties=True, use_missing=True, method='auto'):
    assert x.size == y.size, "Both variables should have the same number of observations."
    n = x.size
    if n < 2:
        return np.nan, np.nan

    rx = rankdata(x, use_missing)
    ry = rankdata(y, use_missing)
    valid = (rx > 0) * (ry > 0)
    rx = rx[valid]
    ry = ry[valid]
    n = np.sum(valid)
    idx = rx.argsort()
    (rx, ry) = (rx[idx], ry[idx])

    C, D = 0, 0
    for i in range(len(ry)-1):
        C += ((ry[i+1:] > ry[i]) * (rx[i+1:] > rx[i])).sum()
        D += ((ry[i+1:] < ry[i]) * (rx[i+1:] > rx[i])).sum()

    xties, corr_x = count_tied_groups(x)
    yties, corr_y = count_tied_groups(y)
    if use_ties:
        denom = np.sqrt((n*(n-1)-corr_x)/2. * (n*(n-1)-corr_y)/2.)
    else:
        denom = n*(n-1)/2.

    if denom == 0.:
        return np.nan, np.nan

    tau = (C-D) / denom

    if method == 'exact' and (len(xties) > 0 or len(yties) > 0):
        raise ValueError("Ties found, exact method cannot be used.")

    if method == 'auto':
        if (len(xties) == 0 and len(yties) == 0) and (n <= 33 or min(C, n*(n-1)/2.0-C) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'

    if len(xties) == 0 and len(yties) == 0 and method == 'exact':
        # Exact p-value, see Maurice G. Kendall, "Rank Correlation Methods" (4th Edition), Charles Griffin & Co., 1970.
        c = int(min(C, (n*(n-1))/2-C))
        if n <= 0:
            raise ValueError
        elif c < 0 or 2*c > n*(n-1):
            raise ValueError
        elif n == 1:
            prob = 1.0
        elif n == 2:
            prob = 1.0
        elif c == 0:
            prob = 2.0/factorial(n)
        elif c == 1:
            prob = 2.0/factorial(n-1)
        else:
            old = [0.0]*(c+1)
            new = [0.0]*(c+1)
            new[0] = 1.0
            new[1] = 1.0
            for j in range(3,n+1):
                old = new[:]
                for k in range(1,min(j,c+1)):
                    new[k] += new[k-1]
                for k in range(j,c+1):
                    new[k] += new[k-1] - old[k-j]
            prob = 2.0*np.sum(np.asarray(new))/factorial(n)
    elif method == 'asymptotic':
        var_s = n*(n-1)*(2*n+5)
        if use_ties:
            v1x, v1y, v2x, v2y = 0, 0, 0, 0
            for k, v in xties.items():
                var_s -= v*k*(k-1)*(2*k+5)*1.
                v1x += v*k*(k-1)
                if n > 2:
                    v2x += v*k*(k-1)*(k-2)
            for k, v in yties.items():
                var_s -= v*k*(k-1)*(2*k+5)*1.
                v1y += v*k*(k-1)
                if n > 2:
                    v2y += v*k*(k-1)*(k-2)
            v1 = v1x * v1y
            v2 = v2x * v2y
            v1 /= 2.*n*(n-1)
            if n > 2:
                v2 /= 9.*n*(n-1)*(n-2)
            else:
                v2 = 0
        else:
            v1 = v2 = 0

        var_s /= 18.
        var_s += (v1 + v2)
        z = (C-D)/np.sqrt(var_s)
        prob = erfc(abs(z)/np.sqrt(2))
    return tau, prob


###############################
##### Auxiliary Functions #####
###############################

@njit
def unique(ar):
    perm = ar.argsort()
    aux = ar[perm]
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    ret = aux[mask]
    idx = np.concatenate((np.nonzero(mask)[0], np.asarray([mask.size])))
    counts = np.diff(idx)
    return ret, counts


@njit
def margins(a):
    n_axis0 = a.shape[1]
    n_axis1 = a.shape[0]
    margsum0 = np.zeros((1, n_axis0))
    margsum1 = np.zeros((n_axis1, 1))
    for i in range(n_axis1):
        margsum1[i, 0] = a[i, :].sum()
    for j in range(n_axis0):
        margsum0[0, j] = a[:, j].sum()
    return margsum1, margsum0


@njit
def expected_freq(observed):
    observed = np.asarray(observed, dtype=np.float64)
    # Create a list of the marginal sums.
    margsum1, margsum0 = margins(observed)
    expected = np.multiply(margsum1, margsum0) / observed.sum()
    return expected


@njit
def rankdata(data, use_missing=True):
    """
        Returns the rank (also known as order statistics) of each data point
        along the given axis.
        If some values are tied, their rank is averaged.
        If some values are masked, their rank is set to 0 if use_missing is False,
        or set to the average rank of the unmasked values if use_missing is True.

        Parameters
        ----------

        data : sequence
            Input data. The data is transformed to a masked array
        use_missing : bool, optional
            Whether the masked values have a rank of 0 (False) or equal to the
            average rank of the unmasked values (True).
    """

    def _rank1d(data, use_missing=False):
        n = data[~np.isnan(data)].size
        rk = np.empty(data.size, dtype=np.float64)
        idx = data.argsort()
        rk[idx[:n]] = np.arange(1, n+1)

        if use_missing:
            rk[idx[n:]] = (n+1)/2.
        else:
            rk[idx[n:]] = -1

        repeats = find_repeats(data.copy())
        for r in repeats[0]:
            condition = (data == r)
            rk[condition] = rk[condition].mean()
        return rk

    if data.ndim > 1:
        return _rank1d(data.ravel(), use_missing).reshape(data.shape)
    else:
        return _rank1d(data, use_missing)


@njit
def find_repeats(arr):
    assert arr.ndim == 1, "find_repeats only accepts 1d arrays."
    if arr.size == 0:
        return np.array([0], arr.dtype), np.array([0], np.int64)

    arr.sort()
    change = np.ones(arr.size, dtype=nb.boolean)
    change[1:] = arr[1:] != arr[:-1]
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + (np.asarray([arr.size]),))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return unique[atleast2], freq[atleast2]


@njit
def tiecorrect(rankvals):
    """
        Tie correction factor for Mann-Whitney U and Kruskal-Wallis H tests.
        Parameters
        ----------
        rankvals : array_like
            A 1-D sequence of ranks.  Typically this will be the array
            returned by `~scipy.stats.rankdata`.
        Returns
        -------
        factor : float
            Correction factor for U or H.

        References
        ----------
        .. [1] Siegel, S. (1956) Nonparametric Statistics for the Behavioral
               Sciences.  New York: McGraw-Hill.
    """

    arr = np.sort(rankvals)
    idx = np.ones(arr.size+1, dtype=nb.boolean)
    idx[1:-1] = arr[1:] != arr[:-1]
    idx = np.nonzero(idx)[0]
    cnt = np.diff(idx).astype(np.float64)

    size = np.float64(arr.size)
    return 1.0 if size < 2 else 1.0 - (cnt**3 - cnt).sum() / (size**3 - size)


@njit
def count_tied_groups(x, use_missing=False):
    """
    Counts the number of tied values.
    Parameters
    ----------
    x : sequence
        Sequence of data on which to counts the ties
    use_missing : bool, optional
        Whether to consider missing values as tied.
    Returns
    -------
    count_tied_groups : dict
        Returns a dictionary (nb of ties: nb of groups).
    Examples
    --------
    >>> from scipy.stats import mstats
    >>> z = [0, 0, 0, 2, 2, 2, 3, 3, 4, 5, 6]
    >>> mstats.count_tied_groups(z)
    {2: 1, 3: 2}
    In the above example, the ties were 0 (3x), 2 (3x) and 3 (2x).
    >>> z = np.ma.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 6])
    >>> mstats.count_tied_groups(z)
    {2: 2, 3: 1}
    >>> z[[1,-1]] = np.ma.masked
    >>> mstats.count_tied_groups(z, use_missing=True)
    {2: 2, 3: 1}
    """
    nmasked = np.sum(np.isnan(x))
    # We need the copy as find_repeats will overwrite the initial data
    data = x.copy()
    (ties, counts) = find_repeats(data)
    nties = {}
    if len(ties):
        for c in np.unique(counts):
            nties[c] = 1
        k, v = find_repeats(counts)
        for i in range(len(k)):
            nties[k[i]] = v[i]

    if nmasked and use_missing:
        if nties.get(nmasked) is not None:
            nties[nmasked] += 1
        else:
            nties[nmasked] = 1

    corr = 0
    for k, v in nties.items():
        corr += v*k*(k-1)

    return nties, corr


@njit(locals={'fac': uint64})
def factorial(n):
    assert n >= 0, "Factorial is only defined for positive numbers."
    if n == 0:
        return 1
    if n <= 20:
        fac = 1
        while n > 0:
            fac *= n
            n -= 1
        return fac
    else:
        return gamma(n+1)
