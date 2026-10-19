"""Rooted homomorphism counts Hom(F, H)[r_1 -> v_1, ..., r_k -> v_k].

``hom_tensor`` returns a numpy array with one axis of length n = |V(H)| per root
(a scalar for unrooted F, a vector for one root, an n x n matrix for two).
H is given by its adjacency matrix A: edge x -> y of F uses A[h(x), h(y)], so
pass a symmetric A for undirected H. A may hold edge weights/multiplicities;
then each hom is weighted by the product of its edge weights.

Algorithm: variable elimination (sum-product) over the vertices of F, keeping
the root vertices free. Cost is O(n^(w+1)) time and O(n^w) memory, where w is
the largest factor arity reached; w = 2 for every quotient of a cycle C_k with
k <= 7 (and for trees w = 1). Dense arrays, so intended for H with up to a few
thousand vertices.

``hom_bruteforce`` enumerates homomorphisms explicitly and is only for tests.
"""
from string import ascii_letters

import numpy as np

from .patterns import ROOTS, root_nodes


def hom_tensor(F, A, max_entries=4e8):
    n = A.shape[0]
    letter = {v: ascii_letters[i] for i, v in enumerate(F.nodes)}
    factors = [((u, v), A) for u, v in F.edges]
    factors += [((v,), np.ones(n, dtype=A.dtype)) for v in F if F.degree(v) == 0]
    free = {v for v in F if F.nodes[v][ROOTS]}
    to_eliminate = set(F.nodes) - free

    def einsum(fs, out_vars):
        spec = ",".join("".join(letter[x] for x in vs) for vs, _ in fs)
        spec += "->" + "".join(letter[x] for x in out_vars)
        return np.einsum(spec, *(t for _, t in fs), optimize="greedy")

    while to_eliminate:
        # Greedy min-degree: eliminate the vertex whose new factor is smallest.
        def scope(x):
            return set().union(*(vs for vs, _ in factors if x in vs)) - {x}
        x = min(to_eliminate, key=lambda x: (len(scope(x)), x))
        sc = sorted(scope(x))
        if n ** len(sc) > max_entries:
            raise MemoryError(f"eliminating {x} needs a factor with n^{len(sc)} entries "
                              f"(n={n}); use a smaller host graph or raise max_entries")
        touching = [f for f in factors if x in f[0]]
        factors = [f for f in factors if x not in f[0]]
        factors.append((tuple(sc), einsum(touching, sc)))
        to_eliminate.remove(x)

    roots = root_nodes(F)
    free_order = sorted(free)
    result = einsum(factors, free_order) if factors else np.array(1, dtype=A.dtype)
    if len(free_order) == len(roots):
        # One root per free vertex: just permute axes into root order.
        return np.transpose(result, [free_order.index(v) for v in roots]) if roots else result
    # Some vertex carries several roots: spread its axis onto a diagonal.
    out_letters = "".join(ascii_letters[26 + i] for i in range(len(roots)))
    delta_specs, deltas = [], []
    for v in free_order:
        rs = sorted(F.nodes[v][ROOTS])
        if len(rs) > 1:
            d = np.zeros((n,) * (len(rs) + 1), dtype=A.dtype)
            d[(np.arange(n),) * (len(rs) + 1)] = 1
            deltas.append(d)
            delta_specs.append(letter[v] + "".join(out_letters[i] for i in rs))
    res_spec = "".join(letter[v] if len(F.nodes[v][ROOTS]) > 1 else out_letters[min(F.nodes[v][ROOTS])]
                       for v in free_order)
    spec = ",".join([res_spec] + delta_specs) + "->" + out_letters
    return np.einsum(spec, result, *deltas)


def sub_tensor(spasm_terms, A, **kw):
    """Sub(G, H)[r = v] = sum_F alpha_F Hom(F, H)[r -> v] for terms from ``spasm(G)``."""
    return sum(float(t.coef) * hom_tensor(t.graph, A, **kw) for t in spasm_terms)


def hom_bruteforce(F, A, injective=False):
    """Same output as hom_tensor, by explicit backtracking (tiny graphs only)."""
    n = A.shape[0]
    roots = root_nodes(F)
    out = np.zeros((n,) * len(roots), dtype=np.int64)
    nodes = list(F.nodes)
    in_edges = {v: [(u, w) for u, w in F.edges if v in (u, w)] for v in nodes}
    h = {}

    def ok(v):
        for u, w in in_edges[v]:
            if u in h and w in h and A[h[u], h[w]] == 0:
                return False
        return True

    def rec(i):
        if i == len(nodes):
            out[tuple(h[r] for r in roots)] += 1
            return
        v = nodes[i]
        used = set(h.values()) if injective else ()
        for img in range(n):
            if img in used:
                continue
            h[v] = img
            if ok(v):
                rec(i + 1)
            del h[v]

    rec(0)
    return out
