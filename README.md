# HomomorphismGNN
Studying graph homomorphisms in Graph Neural Networks.

## Research question

Can we encode information on homomorphisms into the feature updates of a GNN?

If so, can we use this on a real life problem, maybe a dataset?

## Code: spasm + coefficients + rooted hom counts

```
homcount/            small library (numpy/scipy/networkx only)
  patterns.py        patterns with roots: cycle, path, star, make_pattern
  spasm.py           spasm(F) -> quotients with coefficients alpha_F, for 0, 1, 2, ... roots
  hom.py             hom_tensor: dense rooted hom counts (any pattern, small hosts); brute force for tests
  hom_sparse.py      hom_sparse: sparse rooted hom counts (treewidth <= 2, <= 2 roots, large sparse hosts)
tests/               brute-force verification of hom counts and the spasm identity
explore/             exploration scripts (jupytext "percent" format: run as scripts or open as notebooks)
  01_spasm_tour.py       spasm tables for unrooted / anchored / doubly-anchored patterns
  02_aml_edge_counts.py  doubly-rooted counts on every edge of the IBM AML graph, vs. laundering labels
```

The identity implemented is, for a pattern G with roots r and a loop-free host H,

    Sub(G, H)[r = v] = sum_{F in Spasm^r(G)} alpha_F Hom(F, H)[r -> v],
    alpha_F = (-1)^(|V(G)|-|V(F)|) / Aut^r(G) * sum_{rho : G/rho ~ F} prod_{B in rho} (|B|-1)!

(the sum over partitions rho matters: e.g. 5 partitions of C5 give the triangle, hence alpha = 1/2).

### Reproduce (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q tests                      # ~125 tests, a few seconds
python explore\01_spasm_tour.py                # prints spasm tables, writes outputs\spasm_C5_edge_rooted.png
python explore\02_aml_edge_counts.py HI-Small  # ~2-3 min; writes outputs\HI-Small_edge_*.csv*
```

Data: IBM "Transactions for Anti Money Laundering" (Kaggle, `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`);
put `<DATASET>_Trans.csv` and `<DATASET>_Patterns.txt` in `data_raw/`.
To open a script as a notebook: `jupytext --to ipynb explore/02_aml_edge_counts.py`.
