"""Sparse rooted hom counts for large sparse hosts (e.g. transaction graphs).

Same variable elimination as ``hom.hom_tensor`` but with scipy.sparse factors,
so it only supports factors of arity <= 2: patterns of treewidth <= 2 with the
roots kept free (all quotients of C_k for k <= 7, all trees, ...). At most two
roots. Returns a float (no roots), a length-n vector (one root) or an n x n
``scipy.sparse.csr_array`` (two roots).

Cost is driven by the fill of products like A @ A, i.e. by sum_v deg(v)^2, so
remove or down-weight hubs first.
"""
import numpy as np
import scipy.sparse as sp

from .patterns import ROOTS, root_nodes


def hom_sparse(F, A):
    A = sp.csr_array(A, dtype=np.float64)
    n = A.shape[0]
    factors = [((u, v), A) for u, v in F.edges]
    factors += [((v,), np.ones(n)) for v in F if F.degree(v) == 0]
    free = {v for v in F if F.nodes[v][ROOTS]}
    roots = root_nodes(F)
    if len(roots) > 2:
        raise NotImplementedError("hom_sparse supports at most two roots")
    to_eliminate = set(F.nodes) - free

    def scope(x):
        return set().union(*(vs for vs, _ in factors if x in vs)) - {x}

    while to_eliminate:
        x = min(to_eliminate, key=lambda x: (len(scope(x)), x))
        touching = [f for f in factors if x in f[0]]
        factors = [f for f in factors if x not in f[0]]
        g, mats = _collect(x, touching, n)
        others = sorted(mats)
        if len(others) == 0:
            factors.append(((), float(g.sum())))
        elif len(others) == 1:
            factors.append(((others[0],), mats[others[0]].T @ g))
        elif len(others) == 2:
            y, z = others
            factors.append(((y, z), sp.csr_array(mats[y].T @ sp.diags_array(g) @ mats[z])))
        else:
            raise NotImplementedError(f"eliminating {x} needs an arity-{len(others)} factor; "
                                      "use the dense hom_tensor on a smaller graph")
        to_eliminate.remove(x)

    scalar = 1.0
    for vs, t in factors:
        if not vs:
            scalar *= t
    factors = [f for f in factors if f[0]]

    if len(free) == 0:
        return scalar
    if len(free) == 1:
        (v,) = free
        g, _ = _collect(v, factors, n)
        return scalar * (sp.diags_array(g, format="csr") if len(roots) == 2 else g)
    a, b = roots
    ga, mats = _collect(a, [f for f in factors if a in f[0]], n)
    gb, _ = _collect(b, [f for f in factors if f[0] == (b,)], n)
    if b not in mats:
        raise NotImplementedError("roots in different components: result would be dense")
    M = sp.diags_array(ga) @ mats[b] @ sp.diags_array(gb)
    return sp.csr_array(scalar * M)


def _collect(x, touching, n):
    """Multiply the factors touching x: a vector on x and one matrix (x, y) per other var y."""
    g = np.ones(n)
    mats = {}
    for vs, t in touching:
        if len(vs) == 1:
            g = g * t
        else:
            M, y = (t, vs[1]) if vs[0] == x else (t.T, vs[0])
            mats[y] = sp.csr_array(mats[y].multiply(M)) if y in mats else M
    return g, mats
