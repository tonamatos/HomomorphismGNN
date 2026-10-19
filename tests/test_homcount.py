"""Verify hom counting and the (multi-)rooted spasm identity against brute force.

Run from the repo root:  python -m pytest -q tests
"""
from fractions import Fraction

import numpy as np
import pytest

from homcount import (cycle, hom_bruteforce, hom_tensor, make_pattern, path,
                      rooted_automorphisms, spasm, star, sub_tensor)


def random_host(n, p, directed, seed):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p).astype(np.int64)
    np.fill_diagonal(A, 0)
    if not directed:
        A = np.triu(A, 1)
        A = A + A.T
    return A


# (name, pattern) pairs; roots given as node lists.
UNDIRECTED = [
    ("C3", cycle(3)), ("C5", cycle(5)), ("C6", cycle(6)),
    ("C4 rooted", cycle(4, roots=[0])),
    ("C5 rooted", cycle(5, roots=[0])),
    ("P3 rooted end", path(3, roots=[0])),
    ("K4 rooted", make_pattern(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], roots=[0])),
    ("house rooted", make_pattern(5, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)], roots=[2])),
    # doubly rooted
    ("C4 edge-rooted", cycle(4, roots=[0, 1])),
    ("C5 edge-rooted", cycle(5, roots=[0, 1])),
    ("C6 edge-rooted", cycle(6, roots=[0, 1])),
    ("C6 opposite-rooted", cycle(6, roots=[0, 3])),
    ("P2 end-rooted", path(2, roots=[0, 2])),
    ("P3 end-rooted", path(3, roots=[0, 3])),
    ("K4-e diag-rooted", make_pattern(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], roots=[1, 3])),
    # triply rooted
    ("C5 3-rooted", cycle(5, roots=[0, 1, 3])),
]
DIRECTED = [
    ("dC3", cycle(3, directed=True)),
    ("dC4 rooted", cycle(4, roots=[0], directed=True)),
    ("dC4 edge-rooted", cycle(4, roots=[0, 1], directed=True)),
    ("dC5 edge-rooted", cycle(5, roots=[0, 1], directed=True)),
    ("dC4 opposite-rooted", cycle(4, roots=[0, 2], directed=True)),
    ("fan-out center", star(3, roots=[0], directed=True)),
    ("fan-in leaf", star(3, roots=[1], directed=True, out=False)),
    ("dP3 end-rooted", path(3, roots=[0, 3], directed=True)),
    ("scatter-gather", make_pattern(4, [(0, 1), (0, 2), (1, 3), (2, 3)], roots=[0, 3], directed=True)),
]
CASES = [(name, F, False) for name, F in UNDIRECTED] + [(name, F, True) for name, F in DIRECTED]


@pytest.mark.parametrize("name,F,directed", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("seed", [0, 1])
def test_hom_tensor_matches_bruteforce(name, F, directed, seed):
    A = random_host(7, 0.5 if not directed else 0.4, directed, seed)
    assert np.array_equal(hom_tensor(F, A), hom_bruteforce(F, A))


@pytest.mark.parametrize("name,F,directed", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("seed", [0, 1])
def test_spasm_identity(name, F, directed, seed):
    """Inj(F,H)[r->v] = sum_Q inj_coef(Q) Hom(Q,H)[r->v], exactly, at every root tuple."""
    A = random_host(7, 0.5 if not directed else 0.4, directed, seed)
    terms = spasm(F)
    inj = hom_bruteforce(F, A, injective=True)
    rhs = sum(t.inj_coef * hom_tensor(t.graph, A) for t in terms)
    assert np.array_equal(inj, rhs)
    aut = rooted_automorphisms(F)
    assert np.allclose(sub_tensor(terms, A.astype(float)), inj / aut)


def test_distinct_roots_drops_only_diagonal_terms():
    F = cycle(6, roots=[0, 3])
    A = random_host(7, 0.5, False, 3)
    full = sub_tensor(spasm(F), A.astype(float))
    off = sub_tensor(spasm(F, distinct_roots=True), A.astype(float))
    mask = ~np.eye(7, dtype=bool)
    assert np.allclose(full[mask], off[mask])
    assert np.allclose(np.diag(full), 0)


def test_paper_examples():
    # Jin et al. Example 1.3: Sub(C5) = 1/10 Hom(C5) - 1/2 Hom(K3+pendant) + 1/2 Hom(K3)
    assert sorted(t.coef for t in spasm(cycle(5))) == [Fraction(-1, 2), Fraction(1, 10), Fraction(1, 2)]
    # Example 4.4: the anchored spasm of C4 has four elements
    assert len(spasm(cycle(4, roots=[0]))) == 4


def test_cycles_through_an_edge():
    """Doubly-rooted Sub(C_k)[0=u, 1=v] is the number of k-cycles through edge uv."""
    import networkx as nx
    A = random_host(8, 0.5, False, 5)
    H = nx.from_numpy_array(A)
    S = sub_tensor(spasm(cycle(5, roots=[0, 1])), A.astype(float))
    cycles = [c for c in nx.simple_cycles(H, length_bound=5) if len(c) == 5]
    for u, v in H.edges:
        n_uv = sum(1 for c in cycles
                   if any({c[i], c[(i + 1) % 5]} == {u, v} for i in range(5)))
        assert round(S[u, v]) == n_uv


# Excluded: >2 roots, K4 (treewidth 3), and C6 rooted at opposite vertices (one of
# its quotients needs an arity-3 factor once both roots are pinned).
SPARSE_OK = [c for c in CASES if not any(s in c[0] for s in ("3-rooted", "K4", "C6 opposite"))]


@pytest.mark.parametrize("name,F,directed", SPARSE_OK, ids=[c[0] for c in SPARSE_OK])
def test_hom_sparse_matches_dense(name, F, directed):
    from homcount import hom_sparse
    A = random_host(9, 0.4, directed, 7).astype(float)
    for term in spasm(F):
        S = hom_sparse(term.graph, A)
        S = S.toarray() if hasattr(S, "toarray") else S
        assert np.allclose(S, hom_tensor(term.graph, A))
