# Ideas for the research question

> *Can we encode information on homomorphisms into the feature updates of a GNN? Can we use this on a real-life problem?*

**Baseline (Jin et al., ICML 2024):** precompute Hom(F, G) for F in Spasm(H), then attach the counts to the input node features (or to the final readout). The counts never enter the update step itself. "Into the feature updates" could go further than that.

## Candidate ideas

1. **Homomorphism-aware messages (edge level).** Weight or featurize each message u→v with *doubly anchored* counts Hom(F, G)[a↦u, b↦v] for F in a spasm, e.g. how many 5-cycles pass through the edge uv. This puts homomorphisms directly into the update rule. Theory hook: extend the anchored spasm identity (their Thm 4.5) to two anchors and characterize the expressivity gained.
2. **Pattern-based aggregation.** Aggregate over homomorphic images of a rooted pattern F at v rather than over neighbors (standard MPNN = F is a single edge). Theory hook: characterize the homomorphism expressivity of such layers (which graph class's Hom counts they capture), in the style of Zhang et al. 2024 and Caterpillar GNNs.
3. **Inject at every layer, not just the input.** A cheap ablation: does re-injecting the spasm counts into each update help, or does the network already carry them forward?
4. **Densities and the sparse/dense regime.** Replace raw counts (which explode on large graphs) by densities t(F,G) = Hom(F,G)/n^|V(F)|. Hypothesis: in dense graphs the lower-order spasm terms vanish (Sub ≈ Hom/Aut, so the basis adds little), while in sparse graphs such as molecules all connected F have Hom = Θ(n), so the basis matters. Test on synthetic graphs of controlled density, then on real data, then on size generalization (train small, test large).
5. **Minimal basis per architecture.** Inject only the spasm elements the architecture cannot already count (e.g. drop trees for GIN). There is a "weak oddomorphism" criterion the authors commented out of the paper; it characterizes when adding a pattern strictly helps.
6. **Labeled motifs for chemistry (the "real-life" angle).** Labeled spasm bases for functional groups (Sec 4.5 of the paper, never tested).

**Datasets:** ZINC, QM9, ogbg-molhiv (molecules, sparse); Peptides/LRGB (long-range); ogbl-collab (large, denser); BREC (expressivity sanity check).

## Concrete next steps

**Read**
- Bao et al., *Homomorphism Counts as Structural Encodings* (ICLR 2025): the same group's follow-up, so check overlap first.
- Zhang et al., *Beyond Weisfeiler-Lehman* (ICLR 2024): homomorphism expressivity of GNN architectures.
- Barceló et al., *GNNs with Local Graph Parameters* (NeurIPS 2021); NT & Maehara, *Graph Homomorphism Convolution* (ICML 2020): closest prior work on idea 2.
- *When Does More Expressivity Hurt Generalization?* (arXiv 2505.11298): the counter-argument to anticipate.

**Implement**
- Spasm + coefficients + (anchored) hom counting, verified against brute-force Sub counts on random graphs.
- Reproduce GIN on ZINC from `github.com/ejin700/hombasis-gnn` (Base vs Spasm, ≥ 5 seeds) to get a baseline we control.
- Prototype idea 1 or 3 on ZINC, since it is cheap and fast to iterate.

## Questions for the meeting
- Does "into the feature updates" mean inside the layers (ideas 1–3), or is input injection (4–6) also in scope?
- Theory-first (new expressivity result + experiments) or application-first (one domain, strong results)?
- Target venue and timeline (ICLR/NeurIPS/LoG vs. TMLR)?
