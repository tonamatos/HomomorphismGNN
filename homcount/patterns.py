"""Small pattern graphs with optional roots (anchors).

A pattern is a networkx Graph or DiGraph on nodes 0..k-1. Roots are stored as
a node attribute ``roots``: a frozenset of root indices carried by that node.
Root i is the i-th anchor; hom counts get one output axis per root index.
After taking quotients a single node may carry several roots (e.g. {0, 1}).
"""
import networkx as nx

ROOTS = "roots"


def make_pattern(n, edges, roots=(), directed=False):
    """Pattern on nodes 0..n-1; ``roots[i]`` is the node that carries root i."""
    F = nx.DiGraph() if directed else nx.Graph()
    F.add_nodes_from(range(n), **{ROOTS: frozenset()})
    F.add_edges_from(edges)
    for i, v in enumerate(roots):
        F.nodes[v][ROOTS] = F.nodes[v][ROOTS] | {i}
    return F


def cycle(k, roots=(), directed=False):
    """Cycle on k vertices (directed: 0 -> 1 -> ... -> k-1 -> 0)."""
    return make_pattern(k, [(i, (i + 1) % k) for i in range(k)], roots, directed)


def path(n_edges, roots=(), directed=False):
    """Path 0 - 1 - ... - n_edges (directed: left to right)."""
    return make_pattern(n_edges + 1, [(i, i + 1) for i in range(n_edges)], roots, directed)


def star(n_leaves, roots=(), directed=False, out=True):
    """Star with center 0. Directed: fan-out (center -> leaves) if out, else fan-in."""
    edges = [(0, i) if out else (i, 0) for i in range(1, n_leaves + 1)]
    return make_pattern(n_leaves + 1, edges, roots, directed)


def num_roots(F):
    return sum(len(F.nodes[v][ROOTS]) for v in F)


def root_nodes(F):
    """Node carrying each root index, as a list indexed by root."""
    out = [None] * num_roots(F)
    for v in F:
        for i in F.nodes[v][ROOTS]:
            out[i] = v
    return out


def describe(F):
    """Short human-readable description, e.g. 'n=3 E=[(0,1),(1,2)] roots=[0,2]'."""
    edges = ",".join(f"({u},{v})" for u, v in sorted(F.edges()))
    rs = root_nodes(F)
    return f"n={F.number_of_nodes()} E=[{edges}]" + (f" roots={rs}" if rs else "")
