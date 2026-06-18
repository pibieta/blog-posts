# Labor mobility — worker-conditioned transition networks

Project home for an extension of the Mealy, del Rio-Chanona, Farmer (2018) *Job Space* framework. The plan is to move from the paper's *occupation-level* task-similarity network to a *worker-conditioned* mobility model: separate task proximity from transition accessibility, learn occupational and worker embeddings jointly, and use the resulting framework to evaluate retraining and relocation interventions.

The reusable concepts (the $\gamma_{ij}$ similarity, the Job Space network, the scarcity-weighted feature overlap technique, and the task-similarity-vs-transition-accessibility distinction) live in the KB: `~/Projects/knowledge-base/notes/` and `~/Projects/knowledge-base/sources/mealy-rio-chanona-farmer-2018-job-space.md`.

## Working title

*From Job Space to Mobility Space: Learning Worker-Conditioned Labor Transition Networks*

## Research question

Can machine learning separate **occupational task similarity** from **structural transition accessibility** in labor-market networks?

## Research hypothesis

The Mealy et al. task-based Job Space captures the topology of occupational similarity. But observed transition probabilities $P_{ij}$ also encode worker heterogeneity, credential barriers, geography, class background, and unequal access to training and hiring opportunities. A machine-learning framework that conditions on worker covariates can estimate *worker-conditioned transition feasibility*, distinguishing task proximity from structural accessibility — and identify which transitions are technically plausible, which are socially constrained, and which policy interventions could expand feasible mobility for displaced workers.

## Abstract seed

Existing labor-network models show that workers are more likely to transition between occupations with similar work activities. However, observed transition probabilities also encode worker heterogeneity, credential barriers, geography, class background, and unequal access to training and hiring opportunities. This project proposes a machine-learning framework to estimate worker-conditioned transition feasibility, distinguishing task proximity from structural accessibility. By learning occupational and worker embeddings from labor-flow data, the model aims to identify which transitions are technically plausible, which are socially constrained, and which policy interventions could expand feasible mobility for displaced workers.

## ML extension directions

Five concrete directions worth scoping. Order is roughly by ease-of-prototype.

### 1. Learn occupational embeddings

Use observed transitions, work activities, wages, education, geography, and employment flows to learn dense occupation embeddings. The learned space should capture not only task similarity but actual empirical mobility.

Methods:
- node2vec / DeepWalk on the job transition network.
- Graph neural networks on the bipartite occupation × work-activity graph.
- Matrix factorisation of the $T_{ij}$ transition matrix with side information.
- Contrastive learning with positive pairs from observed transitions and negatives from random occupation pairs.

### 2. Model transition probabilities directly

Train a supervised model

$$P(i \to j) = f(\text{task similarity},\ \text{wage gap},\ \text{education gap},\ \text{employment size},\ \text{geography},\ \text{worker features})$$

The decomposition reveals how much predictive power comes from task similarity, education barriers, wage incentives, labor demand, demographic constraints, geography, and prior worker history. Lets us answer: how much of the 91% of variance in $\log P_{ij}$ that $\gamma_{ij}$ doesn't explain is recoverable from observable worker features?

### 3. Estimate individualised transition feasibility

Instead of asking *which occupations are close to occupation $i$?*, ask *which occupations are feasible for worker $w$, given their current occupation, skills, education, location, and resources?*

This is the move from an occupation-level Job Space to a worker-conditioned mobility space. The output is a per-worker feasibility distribution over target occupations, not a global $\gamma$ matrix.

### 4. Separate task proximity from structural access

Estimate two distinct quantities:

$$\text{TaskSimilarity}_{ij} \qquad \text{and} \qquad \text{AccessBarrier}_{w, ij}$$

The first is about work content. The second is about whether a specific worker can actually cross the edge. The two can be parameterised as separate heads of a joint model, with the access head conditioned on worker features and the similarity head purely on occupation features.

This is the operational version of the [task-similarity vs transition-accessibility](../../knowledge-base/notes/task-similarity-vs-transition-accessibility.md) distinction in the KB.

### 5. Counterfactual policy simulations

Once worker-conditioned transition probabilities are learned, simulate interventions:

- Subsidised training (reduces credential barrier).
- Credential relaxation (lowers $c_w$ effective gap for target occupations).
- Relocation assistance (relaxes $g_w$ constraint).
- Apprenticeship programs (reduces experience-gap penalty).
- Job-matching platforms (reduces search friction).
- Automation shocks (displaces workers from high-Frey-Osborne regions).
- Green-transition policies (rebalances brown-job exposure).

Counterfactual question: *which interventions open the most mobility paths for displaced workers?*

## Data sources (probably reusing the paper's)

- O*NET intermediate work activities (332 discrete activities for the $A_{iw}$ matrix).
- US CPS monthly panel for job-to-job transitions (Jan 2010 – Jan 2017 in the original paper; updating to a more recent window is straightforward).
- IPUMS-USA for wages, education, employment, demographics.
- Frey & Osborne (2017) automation susceptibility scores.
- Vona et al. (2017) "brown" job indicators.

Worker-conditioned modeling will need additional data:
- Individual-level transition data with covariates (CPS provides some, but linked employer-employee data from QWI/LEHD would be richer).
- Geography at finer resolution than CPS reports publicly.
- Education-pathway data (credentials, certifications, retraining program participation).

## Status

Project scaffolded 2026-06-17 alongside the KB ingestion of the source paper. Next steps:

- [ ] Pull O*NET intermediate work activities and verify the $A_{iw}$ matrix reproduces the paper's $\gamma_{ij}$ values for a known subset.
- [ ] Replicate the paper's Pearson $\rho = 0.301$ and 9% variance-explained numbers on the same CPS window.
- [ ] Sketch the joint task-similarity / access-barrier model architecture.
- [ ] Identify which CPS covariates can serve as initial worker features ($x_w, c_w, g_w, e_w$).

## Pointers back to the KB

- Source capture: `~/Projects/knowledge-base/sources/mealy-rio-chanona-farmer-2018-job-space.md`
- Atomic notes:
  - `notes/occupational-similarity-measure.md` — the $\gamma_{ij}$ formula and what it measures.
  - `notes/job-space.md` — the network construction and the empirical headline.
  - `notes/scarcity-weighted-feature-overlap.md` — the cross-domain technique.
  - `notes/task-similarity-vs-transition-accessibility.md` — the structural-vs-conditioned split this project operationalises.

When project-specific concepts stabilise (e.g. a particular learned occupational embedding becomes a citable object across multiple project artefacts), distill them back into KB notes — same pattern as the CMI-framework / KB relationship.
