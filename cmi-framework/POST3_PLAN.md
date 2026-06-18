# Post 3 plan — Unique-score-contribution CMI

**Working title (provisional):** *The combine sibling — what one score adds when you already have another.*

**Status:** plan locked 2026-06-10. Ready to be drafted on a subsequent pass.

**Series context.**
- Post 1 (published): *CMI is average log-loss gain.* The empirical equivalence and cross-fitting.
- Post 2 (published, with two closing-sentence fixes still pending): *Cross-target portability.* $I(S^{(1)}; Y^{(2)} \mid Y^{(1)})$ as a portability diagnostic — conditioning on a label.
- Post 3 (this plan): *Unique-score-contribution CMI* — conditioning on a *score*.
- Post 4 (queued): *Partial information decomposition.* The Unq/Red/Syn split that resolves the conflations flagged in posts 2–3.

---

## Premise

Posts 1 and 2 conditioned on labels ($\emptyset$ or $Y^{(1)}$). Post 3 conditions on a *score* — a continuous, model-output object. Same machinery, harder estimator, two readings of the same number.

The decomposition the post hangs on is the chain rule applied to two scores predicting one target:

$$
I\!\left(S^{(1)}, S^{(2)}; Y^{(2)}\right)
\;=\; I\!\left(S^{(2)}; Y^{(2)}\right) \;+\; \underbrace{I\!\left(S^{(1)}; Y^{(2)} \mid S^{(2)}\right)}_{\text{combine}}
\;=\; I\!\left(S^{(1)}; Y^{(2)}\right) \;+\; \underbrace{I\!\left(S^{(2)}; Y^{(2)} \mid S^{(1)}\right)}_{\text{switch}}
$$

One population quantity, two product-decision readings. The combine sibling answers "does adding $S^{(1)}$ on top of $S^{(2)}$ buy anything when both are deployed?" — the ensemble question. The switch sibling answers "does the retrained $S^{(2)}$ add anything beyond the cross-target $S^{(1)}$ we already have?" — the replace question.

This is the same chain-rule identity from post 2 with the conditioning swapped from $Y^{(1)}$ to $S^{(2)}$. The wrinkle: the conditioning set is now continuous, which turns the baseline $\hat{p}_0(Y^{(2)} \mid \cdot)$ from a 2×2 table into a flexible regression on a continuous covariate. Both legs of the CMI difference now carry estimation noise, not just the joint leg.

---

## Locked decisions (do not re-litigate without explicit reopen)

1. **Frame.** Post 3 covers *both siblings as one family* under one chain-rule decomposition, not as two separate posts.
2. **Empirical bed.** Reuse Lending Club. The post-2 pipeline already has $S^{(1)}$ (trained on late-30/60) and $S^{(2)}$ (trained on charge-off) sitting in cache. Post 3 extends the pipeline rather than starting fresh.
3. **PID treatment in post 3.** Conceptual flag only — the combine reading is exact, the switch reading conflates Unq + Syn. Full Unq/Red/Syn machinery is deferred to post 4 with a redundancy-measure choice (BROJA is the leading candidate).
4. **Validation simulation.** Built fresh, separate from the bivariate-probit DGP from post 2 (that one doesn't have synergy in the PID sense). New DGP has a dial-able synergistic component; delivered as a *single figure* (switch CMI inflation as synergy is dialled up), not a full validation panel like post 2.
5. **Post-2 closing-sentence revision.** The "In one paragraph" ↔ "What's next" inconsistency in post 2 gets a small revision pass *after* post 3 lands.
6. **Wyner (1978) citation.** Lemma 3.2 is the chain rule above; Lemma 3.1 is the formal "small CMI = conditional redundancy" claim. Source capture at `~/Projects/knowledge-base/sources/wyner-1978-conditional-mutual-information.md` once the proposal in `outputs/wyner-1978-source-capture.md` lands. Cited lightly in the post's math sidebar.

---

## Section-by-section structure

### 1. Lead — pick up post 2's cliffhanger
Open by quoting (or paraphrasing) post 2's closing promise: *"the actual retraining-decision quantity is the unique score contribution — that's the next post."* Frame the post as paying that debt while also expanding the question — the same chain rule contains two product decisions, not one. Set up the post-2 Lending Club result (56% portability ratio for $S^{(1)}$ vs the retrained $S^{(2)}$) as the empirical hook to return to in section 6.

### 2. Two questions, one decomposition
Present the chain rule. Name the siblings (switch and combine) and tie each to its product decision. Sidebar: short measure-theoretic note that this identity holds for arbitrary continuous $S^{(2)}$ (Wyner 1978, Lemma 3.2) — keep it light, don't bog the section down. The takeaway: the same computation, run on the same data, answers two distinct product questions depending on which baseline you condition against.

### 3. Why conditioning on a score changes the question
The genuinely new technical content. Three sub-points:
- **Baseline auxiliary.** $\hat{p}_0(Y^{(2)} \mid S^{(2)})$ is now a 1D regression on a continuous score — isotonic or 1D GBDT — not a 2×2 table. Both legs of the CMI difference carry estimation noise.
- **Fold count matters more.** With two flexible auxiliaries differenced, plug-in bias is real. $K \geq 5$ folds becomes a meaningful recommendation, not a nice-to-have.
- **Out-of-fold score generation.** The conditioning score $S^{(2)}$ must itself be an out-of-fold prediction on the eval rows, or we leak $Y^{(2)}$ into the conditioning set and the CMI we estimate is a different population quantity. This is the same nuisance-model handling pattern from post 1, but the consequence of getting it wrong is more direct here.

### 4. What the number is — and what it isn't
The PID conceptual flag, in plain prose. The combine reading is exact: any positive CMI tells you the ensemble would do better in log-loss, and the size scales linearly with the log-loss improvement. The switch reading is an upper bound on the true *unique* contribution — if $S^{(1)}$ and $S^{(2)}$ have synergistic interactions (their joint distribution carries information about $Y^{(2)}$ that neither marginal does), the switch CMI absorbs that synergy and overstates the case for retraining. Promise the full Unq/Red/Syn split for post 4. Do not introduce BROJA or any other redundancy measure here.

### 5. Validation simulation — make the synergy visible
Single figure. The synthetic DGP (specified below) lets us dial a synergy knob $\alpha \in [0, 1]$ from "no synergy" to "pure synergy." Plot two curves vs $\alpha$:
- *Switch CMI* $\widehat{I}(S^{(2)}; Y^{(2)} \mid S^{(1)})$.
- *True unique contribution of $S^{(2)}$* — computed analytically in the DGP, since we control the data generation.

At $\alpha = 0$ the two curves coincide. As $\alpha \to 1$ the switch CMI rises while the true Unq stays flat (or rises less). The gap *is* the synergy term. One figure, ~6-7 line caption, no extended commentary — the figure carries the section.

### 6. Real data — Lending Club
Compute both siblings on the cached $S^{(1)}, S^{(2)}$. Report:
- $\widehat{I}(S^{(1)}; Y^{(2)})$ — marginal cross-target (already in post 2).
- $\widehat{I}(S^{(2)}; Y^{(2)})$ — self-information of the retrained score.
- $\widehat{I}(S^{(1)}; Y^{(2)} \mid S^{(2)})$ — combine sibling. *Does the cross-target score add anything to an ensemble already containing the retrained one?*
- $\widehat{I}(S^{(2)}; Y^{(2)} \mid S^{(1)})$ — switch sibling. *Does the retrained score add anything beyond the cross-target one?*

Cross back to post 2's 56% portability ratio: the *switch sibling* is the conditional cousin of the marginal "retrained gain" that ratio gestured at, and it carries the PID caveat that the marginal version didn't. Report both with the appropriate caveat — switch as an upper bound on unique retrained contribution, combine as exact on the ensemble question.

Expectation (based on intuition, not yet computed): the combine sibling will be small and the switch sibling moderate. If it lands that way, the natural reading is "the retrained model captures most of what an ensemble would, with modest non-redundant contribution from the cross-target score" — which is consistent with post 2's portability result but adds a sharper combine-vs-switch dimension.

### 7. What this doesn't tell you
The caveats section. Three items:
- **Calibration sensitivity.** Wyner Theorem 3.4 (conditional DPI) makes CMI calibration-invariant in the population limit, but the cross-fitted estimator inherits whatever miscal the auxiliaries carry. Robustness check: refit $\hat{p}_1$ with isotonic post-processing and verify the CMI estimate doesn't move materially.
- **Variance of differenced CMI's.** Both legs are now random; the difference's variance is larger than either alone. Fold-level SE matters — report it.
- **Feature-policy contamination.** Same caveat as post 2 — Lending Club's `grade`, `sub_grade`, `int_rate` encode LC's own underwriting policy. The scores inherit that. Doesn't invalidate the CMI numbers, but qualifies their interpretation.

### 8. In one paragraph + What's next
One-paragraph recap of the lock decisions: combine sibling is exact, switch sibling is an upper bound, the gap *is* synergy, both can be computed on the same cross-fitted pipeline with the right baseline auxiliary. *What's next:* post 4 takes the gap seriously and walks through the BROJA decomposition on the same Lending Club scores. End with a hook — name one practitioner-relevant question that PID resolves and the chain rule alone can't (probably: "when does an ensemble buy you something beyond what either score's standalone metrics suggest?").

---

## Synthetic DGP design — single-figure validation

**Goal.** A DGP where (a) both scores $S^{(1)}, S^{(2)}$ are well-defined functions of latent features; (b) the true Unq, Red, Syn terms in the PID of $I(S^{(1)}, S^{(2)}; Y)$ are known analytically or computable with arbitrary precision; (c) a single scalar knob $\alpha \in [0, 1]$ moves the system from "no synergy" to "pure synergy" with the marginals held fixed.

**Sketch.** Latent variables $U, V \in \{0, 1\}$ i.i.d. Bernoulli(1/2). Construct scores $S^{(1)} = U + \eta_1$, $S^{(2)} = V + \eta_2$ with $\eta_j \sim \mathcal{N}(0, \sigma^2)$ small noise so the scores are continuous. Target $Y$ is generated from a mixture:

$$
P(Y = 1 \mid U, V) = (1 - \alpha) \cdot \tfrac{1}{2}(U + V) \;+\; \alpha \cdot (U \oplus V)
$$

(coefficients tuned so $P(Y = 1)$ stays at $1/2$ for all $\alpha$). At $\alpha = 0$ the target is an additive function of the latents — no synergy. At $\alpha = 1$ the target is a pure XOR of the latents — pure synergy, neither marginal $S^{(j)}$ is predictive alone.

**Analytical reference.** For each $\alpha$ the joint $(S^{(1)}, S^{(2)}, Y)$ distribution is fully specified, so $I(S^{(j)}; Y)$, $I(S^{(1)}, S^{(2)}; Y)$, and the conditional siblings can be computed by direct integration or Monte Carlo with $N$ large enough that the noise floor is below the signal. The "true Unq" curve to plot against the switch sibling is computable in the same way (with whichever PID definition we commit to in post 4 — for the validation figure here, the natural reference is just "the part of $I(S^{(2)}; Y \mid S^{(1)})$ that doesn't disappear at $\alpha = 0$", which is unambiguous and doesn't pre-commit to BROJA).

**Implementation surface.** New file at `~/Projects/blog-posts/cmi-framework/synergy_sim.py` (parallel to `sim.py` from post 2, but a clean rewrite — the bivariate-probit machinery doesn't carry over). Notebook scaffolding: `synergy_validation.ipynb` produces the single figure. Reuse the existing CMI-estimation helpers from `cross_target_pipeline.py` where applicable (cross-fitting, log-loss-gain). Seed-variation check: at least 3 seeds, report mean ± SE on the figure.

---

## Lending Club empirical plan

**Inputs already in place** (from post 2, in `cross_target_pipeline.py`):
- LC cohort cache in `data/` — pickled DataFrames of features + targets.
- Trained scores $S^{(1)}, S^{(2)}$ generated out-of-fold and saved in the JSON result files.
- The cross-fitting harness with K=5 folds, HistGBM + isotonic auxiliaries.

**New work needed:**
- Add a `compute_unique_contribution` helper that takes $(S_a, S_b, Y)$ and returns $\widehat{I}(S_a; Y \mid S_b)$ via the log-loss-gain identity with both auxiliaries flexible (1D GBDT for $\hat{p}_0(Y \mid S_b)$, 2D GBDT for $\hat{p}_1(Y \mid S_a, S_b)$). Cross-fitted.
- Call it twice, once for each sibling: $\widehat{I}(S^{(1)}; Y^{(2)} \mid S^{(2)})$ and $\widehat{I}(S^{(2)}; Y^{(2)} \mid S^{(1)})$.
- Out-of-fold $S^{(2)}$ on the *eval* rows is critical — verify the pipeline already does this; if not, fix. Same for $S^{(1)}$. (Both should be OOF from the post-2 training.)
- Capture results in `unique_contribution_results.json`, parallel to `cross_target_results*.json`. Seed-variation across 3 seeds for the headline numbers.

**Reporting trio for the post:**
1. Combine CMI $\widehat{I}(S^{(1)}; Y^{(2)} \mid S^{(2)})$ with fold-SE.
2. Switch CMI $\widehat{I}(S^{(2)}; Y^{(2)} \mid S^{(1)})$ with fold-SE.
3. Both normalised by $H(Y^{(2)})$ for unit-free reading.

Connect back to post 2's portability ratio (56%) — the switch sibling is the conditional analogue, with the PID caveat made explicit.

---

## Follow-ups (do *not* block post 3)

- **Post-2 closing-sentence fix.** "In one paragraph" names $I(S^{(2)}; Y^{(2)} \mid S^{(1)})$ (switch); "What's next" describes the combine sibling. Both refer to the same chain-rule family, but with opposite framings. After post 3 lands, light revision of those two paragraphs in `posts/cross-target-portability/index.qmd` to align with the "one family, two siblings" framing this post commits to.
- **KB note promotion.** `notes/cmi-unique-score-contribution.md` is currently `seed` and titled with the combine sibling only. Once post 3 is up, propose an edit (parallel to `outputs/kb-edit-proposal-cmi-cross-target.md`) that: (a) promotes to `developing`; (b) broadens to cover both siblings under the chain-rule framing; (c) adds the LC empirical numbers from post 3; (d) cites Wyner 1978 Lemma 3.2.
- **Sources to land in KB first.** `outputs/wyner-1978-source-capture.md` → `sources/wyner-1978-conditional-mutual-information.md`. And `outputs/kb-edit-proposal-cmi-conditional-mutual-information.md` should be applied to `notes/conditional-mutual-information.md` before post 3 is drafted, so the post can cite the KB note with the Wyner-grounded chain rule already in place.

---

## References (provisional)

- Wyner, A. D. (1978). *A Definition of Conditional Mutual Information for Arbitrary Ensembles.* Information and Control 38, 51–59. — Lemma 3.2 (chain rule), Lemma 3.1 (CMI = 0 iff conditional independence), Theorem 3.4 (conditional DPI; calibration-invariance).
- Pinsker, M. S. (1964). *Information and Information Stability of Random Variables and Processes.* — Background.
- Williams, P. L. & Beer, R. D. (2010). *Nonnegative decomposition of multivariate information.* — Cited only in the post-4 forward pointer. Add to `references.bib` now if it isn't there, to keep the bibliography consistent across the series.
- Internal: post 1 (CMI is log-loss gain), post 2 (cross-target portability).
- KB notes (post-merge): `conditional-mutual-information.md`, `cmi-unique-score-contribution.md`, `cmi-cross-target.md`, `cmi-log-loss-equivalence.md`, `cross-fitting.md`, `cmi-vs-auc.md`.
