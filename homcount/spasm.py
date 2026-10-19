"""Spasm of a (rooted) pattern and the coefficients of its homomorphism basis.

For a pattern G with roots r_1..r_k and a loop-free host H,

    Inj(G, H)[r -> v] = sum_rho  mu(bot, rho) * Hom(G/rho, H)[r -> v]
    mu(bot, rho)      = prod_{B in rho} (-1)^(|B|-1) (|B|-1)!
    Sub(G, H)[r = v]  = Inj(G, H)[r -> v] / Aut^r(G)

where rho ranges over partitions of V(G) whose blocks are independent sets
(otherwise G/rho has a loop and Hom into a loop-free H is 0), and Aut^r(G) is
the number of automorphisms fixing every root. Grouping the partitions by the
rooted isomorphism type F of G/rho gives

    Sub(G, H)[r = v] = sum_{F in Spasm^r(G)} alpha_F * Hom(F, H)[r -> v]
    alpha_F = (-1)^(|V(G)|-|V(F)|) / Aut^r(G) * sum_{rho: G/rho ~ F} prod_B (|B|-1)!

Note the *sum* over partitions: each partition contributes its own product.
k = 0 is Curticapean-Dell-Marx, k = 1 is Thm 4.5 of Jin et al. (ICML 2024),
k = 2 is the doubly-rooted version. Nothing in the derivation depends on k,
nor on whether G is directed.

If two roots end up in the same block, F carries both roots on one vertex and
Hom(F, H)[r_1 -> u, r_2 -> v] = 0 whenever u != v. Pass ``distinct_roots=True``
to drop those terms when you only evaluate at tuples of distinct host vertices.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import factorial

import networkx as nx
from networkx.algorithms import isomorphism as iso

from .patterns import ROOTS


@dataclass
class SpasmTerm:
    graph: nx.Graph      # quotient F, nodes 0..|V(F)|-1, roots as node attribute
    coef: Fraction       # alpha_F: Sub = sum alpha_F Hom(F)
    inj_coef: int        # alpha_F * Aut^r(G): Inj = sum inj_coef Hom(F)
    n_partitions: int    # number of partitions rho with G/rho ~ F


def _node_match(a, b):
    return a[ROOTS] == b[ROOTS]


def _matcher(G1, G2):
    cls = iso.DiGraphMatcher if G1.is_directed() else iso.GraphMatcher
    return cls(G1, G2, node_match=_node_match)


def rooted_automorphisms(G):
    """Number of automorphisms of G fixing every root."""
    return sum(1 for _ in _matcher(G, G).isomorphisms_iter())


def rooted_isomorphic(G1, G2):
    return _matcher(G1, G2).is_isomorphic()


def independent_partitions(G):
    """All partitions of V(G) into independent sets (as lists of lists)."""
    nodes = list(G.nodes)
    adj = {v: set(G.successors(v)) | set(G.predecessors(v)) if G.is_directed()
           else set(G.neighbors(v)) for v in nodes}

    def rec(i, blocks):
        if i == len(nodes):
            yield [list(b) for b in blocks]
            return
        v = nodes[i]
        for b in blocks:
            if not (adj[v] & b):
                b.add(v)
                yield from rec(i + 1, blocks)
                b.remove(v)
        blocks.append({v})
        yield from rec(i + 1, blocks)
        blocks.pop()

    yield from rec(0, [])


def quotient(G, blocks):
    """G/rho with nodes 0..len(blocks)-1. Assumes blocks are independent sets."""
    block_of = {v: i for i, B in enumerate(blocks) for v in B}
    Q = G.__class__()
    for i, B in enumerate(blocks):
        Q.add_node(i, **{ROOTS: frozenset().union(*(G.nodes[v][ROOTS] for v in B))})
    Q.add_edges_from((block_of[u], block_of[v]) for u, v in G.edges)
    return Q


def _invariant(Q):
    if Q.is_directed():
        sig = sorted((tuple(sorted(Q.nodes[v][ROOTS])), Q.in_degree(v), Q.out_degree(v)) for v in Q)
    else:
        sig = sorted((tuple(sorted(Q.nodes[v][ROOTS])), Q.degree(v)) for v in Q)
    return (Q.number_of_nodes(), Q.number_of_edges(), tuple(sig))


def spasm(G, distinct_roots=False):
    """Rooted spasm of G with coefficients, largest quotients first."""
    aut = rooted_automorphisms(G)
    n = G.number_of_nodes()
    buckets = {}  # invariant -> list of [representative, weight, n_partitions]
    for blocks in independent_partitions(G):
        Q = quotient(G, blocks)
        if distinct_roots and any(len(Q.nodes[v][ROOTS]) > 1 for v in Q):
            continue
        weight = 1
        for B in blocks:
            weight *= factorial(len(B) - 1)
        for entry in buckets.setdefault(_invariant(Q), []):
            if rooted_isomorphic(entry[0], Q):
                entry[1] += weight
                entry[2] += 1
                break
        else:
            buckets[_invariant(Q)].append([Q, weight, 1])

    terms = []
    for entries in buckets.values():
        for Q, weight, count in entries:
            sign = (-1) ** (n - Q.number_of_nodes())
            terms.append(SpasmTerm(Q, Fraction(sign * weight, aut), sign * weight, count))
    terms.sort(key=lambda t: (-t.graph.number_of_nodes(), -t.graph.number_of_edges()))
    return terms
