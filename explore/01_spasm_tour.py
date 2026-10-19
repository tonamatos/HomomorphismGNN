# %% [markdown]
# # 01 — Spasm tour: unrooted, anchored and doubly-anchored
#
# Prints the homomorphism basis (quotients + coefficients) of a few patterns and
# checks the identity Sub = sum_F alpha_F Hom(F) numerically on a random graph.
# Run from the repo root:  python explore/01_spasm_tour.py
# (or open as a notebook via jupytext). Figures go to outputs/.

# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from homcount import (ROOTS, cycle, describe, hom_bruteforce, path, rooted_automorphisms,
                      spasm, sub_tensor)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def show(name, F, **kw):
    terms = spasm(F, **kw)
    print(f"\n{name}: |Spasm| = {len(terms)}, Aut^r = {rooted_automorphisms(F)}")
    for t in terms:
        print(f"  {str(t.coef):>8}  ({t.n_partitions:>3} partitions)  {describe(t.graph)}")
    return terms


# %% [markdown]
# ## Unrooted and singly-rooted (reproduces Jin et al. Ex. 1.3 and 4.4)

# %%
show("C5", cycle(5))
show("C4 anchored", cycle(4, roots=[0]))

# %% [markdown]
# ## Doubly-rooted: roots on an edge (idea 1: edge-level messages)
#
# Sub(C_k)[0=u, 1=v] is the number of k-cycles through the edge uv. Since the
# roots are adjacent they can never be merged, so the spasm only contains
# quotients with two distinct roots.

# %%
spasm_sizes = {}
for k in range(3, 9):
    terms = spasm(cycle(k, roots=[0, 1]))
    spasm_sizes[k] = (len(spasm(cycle(k))), len(spasm(cycle(k, roots=[0]))), len(terms))
print("k : |Spasm(C_k)|  |Spasm^a(C_k)|  |Spasm^ab(C_k)| (ab = edge)")
for k, s in spasm_sizes.items():
    print(f"{k} : {s[0]:>6} {s[1]:>14} {s[2]:>16}")

c5_edge = show("C5 edge-rooted", cycle(5, roots=[0, 1]))
show("directed C4 edge-rooted", cycle(4, roots=[0, 1], directed=True))

# %% [markdown]
# ## Doubly-rooted at non-adjacent vertices
#
# Now the two roots can fall into the same block. Those quotients carry both
# roots on one vertex and vanish off the diagonal (u != v); `distinct_roots=True`
# drops them.

# %%
show("P2 end-rooted (all)", path(2, roots=[0, 2]))
show("P2 end-rooted (distinct roots)", path(2, roots=[0, 2]), distinct_roots=True)
show("C6 opposite-rooted (distinct roots)", cycle(6, roots=[0, 3]), distinct_roots=True)

# %% [markdown]
# ## Numerical check on a random graph

# %%
rng = np.random.default_rng(0)
A = (rng.random((10, 10)) < 0.4).astype(float)
A = np.triu(A, 1); A = A + A.T
F = cycle(5, roots=[0, 1])
lhs = hom_bruteforce(F, A, injective=True) / rooted_automorphisms(F)
rhs = sub_tensor(spasm(F), A)
print("max |Sub_bruteforce - sum alpha Hom| =", np.abs(lhs - rhs).max())


# %% [markdown]
# ## Figure: doubly-rooted spasm of C5

# %%
def draw_terms(terms, title, fname):
    fig, axes = plt.subplots(1, len(terms), figsize=(2.2 * len(terms), 2.6))
    for ax, t in zip(np.atleast_1d(axes), terms):
        Q = t.graph
        pos = nx.kamada_kawai_layout(Q) if Q.number_of_nodes() > 2 else nx.spring_layout(Q, seed=0)
        colors = ["tab:red" if Q.nodes[v][ROOTS] else "lightgray" for v in Q]
        labels = {v: ",".join("ab"[i] for i in sorted(Q.nodes[v][ROOTS])) for v in Q}
        nx.draw(Q, pos, ax=ax, node_color=colors, labels=labels, node_size=260,
                arrows=Q.is_directed())
        ax.set_title(f"$\\alpha$ = {t.coef}", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    print("saved", OUT / fname)


draw_terms(c5_edge, "Spasm$^{ab}(C_5)$, roots a,b on an edge", "spasm_C5_edge_rooted.png")
