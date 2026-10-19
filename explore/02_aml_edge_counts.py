# %% [markdown]
# # 02 — Doubly-rooted hom counts on IBM AML transactions (edge level)
#
# For every account pair (u -> v) with at least one transaction we compute
#   * Hom(F, H)[a -> u, b -> v] for every F in the doubly-rooted spasm of
#     directed cycles dC_k (k = 2..6), undirected cycles C_k (k = 3..6) and
#     directed u -> ... -> v paths (the "scatter-gather"/"stack" shape),
#   * the corresponding Sub count = sum_F alpha_F Hom(F) (e.g. number of
#     directed k-cycles through the edge u -> v),
# and check how well each single feature separates laundering edges (AUC).
#
# Host graph H: accounts (bank + account id) as nodes, a simple directed edge
# u -> v if there is any transaction u -> v, self-loops dropped (the identity
# needs a loop-free host). Accounts with degree > DEG_CAP are removed: a few
# bank-internal hubs (out-degree up to ~14k) otherwise dominate every count.
#
# Run from the repo root:  python explore/02_aml_edge_counts.py [DATASET]
# DATASET defaults to HI-Small; expects data_raw/<DATASET>_Trans.csv and _Patterns.txt.

# %%
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from homcount import cycle, hom_sparse, path, spasm

DATASET = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "HI-Small"
DEG_CAP = 100
DATA = ROOT / "data_raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Edge-rooted patterns: (name, pattern, distinct_roots, use directed host?)
PATTERNS = (
    [(f"dC{k}", cycle(k, roots=[0, 1], directed=True), False, True) for k in range(2, 7)]
    + [(f"C{k}", cycle(k, roots=[0, 1]), False, False) for k in range(3, 7)]
    + [(f"dP{k}", path(k, roots=[0, k], directed=True), True, True) for k in (2, 3)]
)

# %% [markdown]
# ## Load transactions and collapse to a simple directed account graph

# %%
t0 = time.time()
tx = pd.read_csv(DATA / f"{DATASET}_Trans.csv", dtype=str, usecols=[1, 2, 3, 4, 10])
tx.columns = ["from_bank", "from_acct", "to_bank", "to_acct", "y"]
tx["src"] = tx.from_bank + "_" + tx.from_acct
tx["dst"] = tx.to_bank + "_" + tx.to_acct
tx["y"] = tx.y.astype(np.int8)
print(f"{len(tx):,} transactions, {tx.y.sum():,} laundering ({time.time() - t0:.0f}s)")

tx = tx[tx.src != tx.dst]
edges = tx.groupby(["src", "dst"]).agg(n_tx=("y", "size"), y=("y", "max")).reset_index()
del tx
accounts = pd.Index(pd.unique(np.concatenate([edges.src.values, edges.dst.values])))
edges["u"] = accounts.get_indexer(edges.src)
edges["v"] = accounts.get_indexer(edges.dst)
n = len(accounts)
print(f"{n:,} accounts, {len(edges):,} directed edges, {edges.y.sum():,} laundering edges")

# %% [markdown]
# ## Laundering typologies from the Patterns file

# %%
rows = []
typology = None
for line in open(DATA / f"{DATASET}_Patterns.txt"):
    line = line.strip()
    if line.startswith("BEGIN"):
        typology = line.split("- ", 1)[1].split(":")[0].strip()
    elif line.startswith("END"):
        typology = None
    elif line and typology:
        f = line.split(",")
        rows.append((f[1] + "_" + f[2], f[3] + "_" + f[4], typology))
pat = pd.DataFrame(rows, columns=["src", "dst", "typology"]).drop_duplicates(["src", "dst"])
edges = edges.merge(pat, on=["src", "dst"], how="left")
edges["typology"] = edges.typology.fillna("")
print(edges[edges.typology != ""].typology.value_counts())

# %% [markdown]
# ## Remove hubs

# %%
A = sp.csr_array((np.ones(len(edges)), (edges.u.values, edges.v.values)), shape=(n, n))
A_und = sp.csr_array(((A + A.T) > 0).astype(np.float64))
deg = np.asarray(A_und.sum(axis=1)).ravel()
keep = deg <= DEG_CAP
D = sp.diags_array(keep.astype(np.float64))
A = sp.csr_array(D @ A @ D)
A_und = sp.csr_array(D @ A_und @ D)
edges = edges[keep[edges.u] & keep[edges.v]].reset_index(drop=True)
print(f"removed {(~keep).sum()} accounts with degree > {DEG_CAP}; "
      f"{len(edges):,} edges and {edges.y.sum():,} laundering edges remain")

# %% [markdown]
# ## Doubly-rooted hom counts and Sub counts at every edge

# %%
u, v = edges.u.values, edges.v.values
features = {}
for name, F, distinct, directed in PATTERNS:
    t0 = time.time()
    terms = spasm(F, distinct_roots=distinct)
    host = A if directed else A_und
    sub = np.zeros(len(edges))
    for j, t in enumerate(terms):
        M = sp.csr_array(hom_sparse(t.graph, host))
        vals = np.asarray(M[u, v]).ravel()
        features[f"{name}_hom{j}"] = vals
        sub += float(t.coef) * vals
    features[f"{name}_sub"] = np.rint(sub)
    print(f"{name}: {len(terms)} spasm terms, {time.time() - t0:.1f}s, "
          f"edges with >=1 {name}-subgraph: {(features[f'{name}_sub'] > 0).sum():,}")

# Baselines: plain degrees of the endpoints (after hub removal). Many hom
# counts are dominated by degrees, so they must beat these to be interesting.
out_deg, in_deg = np.asarray(A.sum(axis=1)).ravel(), np.asarray(A.sum(axis=0)).ravel()
und_deg = np.asarray(A_und.sum(axis=1)).ravel()
features.update({"base_outdeg_u": out_deg[u], "base_indeg_u": in_deg[u], "base_outdeg_v": out_deg[v],
                 "base_indeg_v": in_deg[v], "base_deg_u_x_deg_v": und_deg[u] * und_deg[v]})
feat = pd.DataFrame(features)
assert (feat.filter(like="_sub") >= 0).all().all(), "Sub counts must be non-negative"

# %% [markdown]
# ## How well does each feature alone separate laundering edges?
#
# AUC of the raw feature (0.5 = useless; ties are common since most counts are 0).

# %%
def auc(score, y):
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


y = edges.y.values.astype(bool)
summary = pd.DataFrame({
    "auc": {c: auc(feat[c].values, y) for c in feat},
    "nonzero_laund": (feat[y] > 0).mean(),
    "nonzero_normal": (feat[~y] > 0).mean(),
    "mean_laund": feat[y].mean(),
    "mean_normal": feat[~y].mean(),
})
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
print(summary.loc[[c for c in feat if c.endswith("_sub") or c.startswith("base_")]].round(4))
print("\nTop 15 features by |AUC - 0.5|:")
print(summary.reindex((summary.auc - 0.5).abs().sort_values(ascending=False).index).head(15).round(4))

# %% [markdown]
# ## Per typology: fraction of edges lying on at least one pattern copy

# %%
sub_cols = [c for c in feat if c.endswith("_sub")]
# "(not in Patterns)" = labelled laundering but not listed in the Patterns file.
group = np.where(y, edges.typology.replace("", "(not in Patterns)"), "normal")
by_type = (feat[sub_cols] > 0).groupby(group).mean()
by_type["n_edges"] = pd.Series(group).value_counts()
print(by_type.T.round(3))

# %%
summary.to_csv(OUT / f"{DATASET}_edge_feature_summary.csv")
pd.concat([edges[["src", "dst", "n_tx", "y", "typology"]], feat], axis=1).to_csv(
    OUT / f"{DATASET}_edge_features.csv.gz", index=False)
print("saved to", OUT)
