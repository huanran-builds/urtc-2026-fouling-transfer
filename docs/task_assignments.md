# URTC 2026 — Task Assignments

**Updated August 10, 2026. Check-in Sunday August 10, 9pm.**

---

## The one sentence

We're building a model that predicts how well a nanoparticle kills a given bacteria, and testing whether it can be adapted to work on a source paper it has never seen.

## The longer version

Chen et al. (2026, *Cell Reports Physical Science*) built a dataset by pulling 342 nanoparticle results out of 65 published papers. Each row is one nanoparticle tested against one bacterial species. The outcome is MIC, the concentration needed to stop bacterial growth. They trained a model to classify particles as strong, moderate, or weak, and reported 79% accuracy.

They split train and test randomly. One source paper contributes between 5 and 18 rows, so rows from the same paper land on both sides of the split. The model learns "this is that paper again" rather than learning chemistry. When we resplit so no paper appears on both sides, accuracy drops from **0.728 to 0.316**. The majority-class baseline is **0.418**, so it does worse than guessing.

That matters because a model that cannot handle an unseen paper is useless to the person who would want to use it. If you are designing a new particle, you *are* a new paper.

Differences between papers are a **batch effect**, the same problem genomics has when data comes from different labs. Dr. Jun (Edward's dad, biostatistician at UTHealth) confirmed the approach: we may use a new paper's input features to estimate a correction, but never its labels. That is **semi-supervised learning**. A crude version, standardizing each paper's features within that paper, already moved us from 0.316 to 0.345.

**Deliverable:** poster at IEEE MIT URTC, October 9 to 11, MIT.

**Working title:** Unlabeled Target-Study Adaptation for Predicting Nanoparticle Antibacterial Activity Across Source Publications

---

## Where we sit relative to published work

Three papers have landed near this. Read this section before you assume anything we find is new.

**Li et al. (2026), *Green Chemistry* 28:9804-9821, DOI 10.1039/d6gc01077f.** Assembled a literature-derived chemistry dataset and showed random splitting inflates performance versus source-publication-grouped validation: R² above 0.91 random against 0.441 grouped. They stopped at diagnosis. Their recommendation is that the field collect better-standardized data going forward, which does nothing for datasets that already exist.

**Xu et al. (2026), *npj Computational Materials* 12:191, DOI 10.1038/s41524-026-02065-2.** Built a literature-extracted catalysis dataset from 160 papers, treated source DOI as the batch, and benchmarked ComBat and RUV-PC. Their predictive stage still uses an 80/20 random split; the DOI grouping appears only in their causal-analysis framework. They never test whether correction improves prediction on a completely unseen paper.

**Austin et al. (2025), *Nature Microbiology* 10:897-911, DOI 10.1038/s41564-025-01954-4.** DEBIAS-M. Leave-one-study-out adaptation using a held-out study's features but not its labels, with demonstrated improvement in cross-study prediction. Closest match to our setup, but their inputs are pooled raw microbiome datasets, not a table hand-extracted from published results.

**What is left.** The problem is documented, batch correction on literature-extracted data is established, and unlabeled target adaptation is demonstrated. Nobody has combined them: unlabeled adaptation to an unseen *source publication* in a *literature-extracted* dataset, measured by whether predictive accuracy recovers. That is our question.

**Be honest about the size of this.** It is a narrow methodological slice, not a broad claim. That is enough for a URTC poster, where the bar is competence and specificity rather than novelty to the field. It is not enough to claim we invented something.

---

## Rules for everyone

1. **Every number you report is committed to the repo with the script or document that produced it.** No numbers in chat without something behind them.
2. **Use Codex freely, but you own whether the output is correct.** Codex will hand you a script that runs clean and returns a wrong number. Every task below has a verification block. Check against it before you post anything.
3. **Every citation must resolve.** Click the DOI. We got burned once on bad references and it nearly cost us Dr. Jun.
4. **Branch, then PR. Do not push to main.** Branch names: `ezra/mbc`, `matthew/debias`, `stephen/lit`, `edward/audit`. Push the branch, open a pull request, Henry reviews and merges. Commit scripts and their output CSVs together.
5. **If you are blocked, say so and name the person who can unblock you.** "Blocked" with no name is not a report.

## Before you start, 30 minutes

- Read the paper's abstract and the Methods sections on data collection and model building: `cell.com/cell-reports-physical-science/fulltext/S2666-3864(26)00317-6`
- Read `scripts/01_grouped_cv.py`. It is commented. That is the core analysis.
- Read the README.

**Ezra and Matthew also need Python running.** Fresh clone into a new folder, do not pull into an old copy of this repo.

macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install libomp
python scripts/01_grouped_cv.py
```

Windows (skip the brew line entirely, it is macOS only):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/01_grouped_cv.py
```

That last command must print **random 0.728, grouped 0.316, corrected 0.345**. **Tolerance is 0.02 on any single metric.** Small platform differences (ARM Mac vs x86 Windows) are expected in XGBoost even with a fixed seed. If you differ by more than 0.02, stop and tell Henry. Everything downstream assumes we are all running the same pipeline.

---

# MATTHEW — Cross-study adaptation

**Why you.** Batch correction and domain adaptation come out of genomics. DEBIAS-M is a microbiome method. Your comp bio background puts you closest to this literature, and this task is the project's actual contribution.

**You own:** making prediction on an unseen source paper work.

### What changed

DEBIAS-M was evaluated and found structurally incompatible with this table: it
requires non-negative compositional abundance features that are normalized to
sum to one, and it fits its own linear classifier. Our data mix physical
measurements and categorical variables, so that transformation would not be
scientifically interpretable. Document this finding in
`docs/method_compatibility.md`; it is not an empirical DEBIAS-M result.

Use **CORAL** (CORrelation ALignment), implemented in `scripts/05_adaptation.py`.
CORAL aligns source and unlabeled target feature covariances and is appropriate
for general tabular feature vectors.

### Step 1. Reproduce the baseline (Monday)

Run `scripts/01_grouped_cv.py`. Confirm grouped = 0.316. If you get something else, your environment differs and every comparison after this is meaningless. Fix that first.

### Step 2. Read (Monday)

Sun, Feng, and Saenko 2016, DOI 10.1609/aaai.v30i1.10306. Read the CORAL
method and its source-to-unlabeled-target covariance alignment. Also read the
DEBIAS-M compatibility note so its exclusion is stated precisely.

The key idea: train on studies 1 through N-1, hold out study N entirely, use study N's **features** to learn correction factors, never touch study N's labels, then predict.

### Step 3. Implement (Monday to Tuesday)

Create `scripts/05_adaptation.py`. Copy the pipeline structure from `01_grouped_cv.py`.

- `Ref` (forward-filled) is the batch or study label
- Fit CORAL **inside** each CV fold, refit every time
- Evaluate with `GroupKFold(5)` on `Ref`
- The held-out fold's features may be used for correction. Its labels may not. Assert this in code.

### Step 4. Report (Tuesday)

Accuracy and macro F1 for: uncorrected, within-paper standardized, and CORAL.
Report each score relative to the majority-class baseline as well as raw
accuracy.

**Verification. Do not report until all five pass:**

| Check | Expected | If it fails |
| --- | --- | --- |
| Uncorrected grouped accuracy | **0.316** | Your pipeline differs from ours. Fix before comparing. |
| CORAL grouped accuracy | 0.30 to 0.50 is plausible | Above 0.55 means labels may have leaked. Confirm `y` never enters CORAL. |
| Shuffled-label control (`scripts/03_permutation.py`) | ~0.32 random, ~0.41 grouped, both near the 0.418 baseline | Higher means something is leaking |
| Papers with a single row | Handled without crashing | Covariance and variance are undefined for n=1. Fall back to centering. |
| Per-fold accuracy reported | Five numbers, not just a mean | |

**Report per-fold accuracy, not only the mean.** If adaptation helps on some papers and hurts on others, that is a more honest and more interesting result than the average, and it tells us when adaptation works.

**Done when:** `scripts/05_adaptation.py`, `docs/method_compatibility.md`, and
their output CSVs are committed on branch `matthew/debias`, PR opened, and you
have posted the numbers with the verification checks confirmed.

**Due Tuesday August 12.** Not blocking the proposal. If you need an extra day, flag it Sunday rather than going quiet.

---

# EZRA — The bacterial side and the MBC replication

**Why you.** MIC and MBC are clinical microbiology endpoints, and half the feature set is bacterial: species, Gram stain, motility, oxygen requirement, cell shape, arrangement. That is your area. You also own the "why does this matter" argument, which is an AMR argument.

**You own:** the bacteria half of the model, and the significance framing.

### Step 1. Replicate on MBC (Friday to Saturday)

MIC is the concentration that stops growth. MBC is the concentration that kills. Different endpoint, different file, same structural problem.

- File: `data/external/chen2026_nanoparticles/Nanoparticles_MBC.csv`, 133 rows, 24 papers
- Copy `scripts/01_grouped_cv.py` to `scripts/04_mbc.py`, change the path
- **The schemas differ.** The Gram column has a different header in the MBC file. Print `list(df.columns)` first, then rename inside the script. Do not edit the raw CSV.
- There is no `MBC_class` column. Create it by binning `MBC (µg/mL)` with the same clinical breakpoints Chen used for MIC: strong ≤ 10, moderate > 10 and ≤ 100, weak > 100
- Use all 133 raw rows. If the script drops any, log which and why, and report both raw and analyzed counts. The paper says 132 processed; reconciling that is Edward's audit, not yours.
- Same full feature set as the MIC baseline, including `duration` and `temperature`
- Add an assertion that no paper appears in both train and test. That check catches the failure mode where `groups=` silently does not reach the splitter.

Report: random accuracy, grouped accuracy, majority baseline, class counts, fold-level class support, unique papers after `Ref` forward-fill.

**Verification:**

| Check | Expected | If it fails |
| --- | --- | --- |
| Rows, papers | 133, 24 | If each paper appears exactly once, you forgot `df['Ref'] = df['Ref'].ffill()` |
| Random accuracy | 0.55 to 0.80 | |
| Grouped accuracy minus majority baseline | Below +0.10. Compare to baseline, not to an absolute range, since class balance differs between files | If grouped ≈ random, `groups=` is not reaching GroupKFold |
| Class counts | Report them | With 133 rows a class may be very small. Say so. |
| No-overlap assertion | Passes | |

**Why it matters:** one dataset with a result is an anecdote. Two is a pattern.

### Step 2. Bacterial feature analysis (Saturday to Sunday)

New script, `scripts/06_species_analysis.py`.

- Report both **raw** and **normalized** species names, so the normalization is auditable. The CSV has trailing spaces, abbreviations, and strain labels.
- **Flag non-bacterial organisms separately.** *Candida albicans* is a fungus. Report how many rows are non-bacterial. If it is more than a handful, that is a real limitation: a model described as antibacterial would be partly trained on antifungal data.
- Confirm Chen's claim that *E. coli*, *S. aureus*, and *P. aeruginosa* dominate. Give exact counts.
- **The real question:** does the model do better on well-represented species than rare ones? Fit **one** MIC grouped-CV model, then stratify performance using its out-of-fold predictions. Do not train separate models on the rare subset; there is almost no data there.
- Buckets: species with more than 20 rows, 5 to 20 rows, fewer than 5. Report all three. Species with exactly 5 go in the middle bucket. Show the full distribution.

If the model only works on three species, that is a hard limit on the whole thing and we state it before a reviewer finds it.

### Step 3. Significance paragraph (Sunday)

150 words of prose in your own words, `docs/significance.md`. References do not count toward the limit.

- Antimicrobial resistance is outpacing new antibiotics, which is why nanoparticle antibacterials are being pursued
- MIC and MBC are the standard clinical endpoints, so a model predicting them predicts something clinicians actually use
- A model that only works on the papers it was trained on cannot support designing anything new

At least one real, resolving citation per point. This becomes the opening of the poster.

**Done when:** branch `ezra/mbc`, both scripts committed with their output CSVs, `docs/significance.md` committed, PR opened, numbers posted with verification confirmed.

**Due Sunday August 10.**

---

# EDWARD — The materials side and the dataset audit

**Why you.** Half the feature set is materials: composition, size, shape, synthesis route, surface charge, plus 20 elemental descriptors from the Magpie database. You are the only one who can judge whether those descriptors are physically sensible. You also work on surface coatings, and everything here is a surface interacting with a biological medium.

**You own:** the nanoparticle half of the feature set, and data integrity.

### Step 1. Dataset audit (Friday to Saturday)

Three items. Each needs evidence: a page number, a DOI, or a row count.

**1a.** The Cell Reports abstract says **98 source studies**. The MIC file has 65 unique refs and MBC has 24. That is at most 89, fewer if they overlap. Reconcile it or document the gap. Check whether the two files share references.

**1b.** In the MIC file, the ZnS and ZnO-TOP rows share one DOI after forward-fill, 18 rows total. Open `10.1002/cbdv.202400724` and confirm that paper tested **both** materials. If it only covers ZnS, our forward-fill is wrong and our study counts change.

**1c.** Chen et al. **2024** (`10.1186/s12951-024-02974-8`, a different paper, on nanostructured surfaces) states that surface roughness was "approximately 90% or more missing values and was therefore excluded." Their own Table S5 reports it in **90 of 291 rows**, which is 69%. Open the supplementary and count it yourself.

Deliver `docs/data_audit.md`.

### Step 2. Materials feature audit (Saturday to Sunday)

The MIC file has 22 `MagpieData` columns: mean and range of electronegativity, melting temperature, atomic radius, atomic volume, covalent radius, thermal conductivity, density, fusion enthalpy, periodic row, periodic column.

- These are computed from elemental composition using the `matminer` library. For a nanoparticle, **do they mean anything?** The mean electronegativity of ZnO tells you about zinc and oxygen atoms. It tells you nothing about surface area, crystal facet, or capping agent.
- Chen et al. say as much themselves. They call these "compositional or materials-informatics proxy descriptors" as opposed to "mechanism-proximal descriptors," and warn they should be "interpreted more cautiously." Find and read that passage.
- **Your question:** which of these 22 features are doing real work, and which are just encoding "this is silver"? If most are proxies for material identity, and material identity is nearly constant within a paper, the Magpie block may be a substantial part of why the model memorizes papers.

Write `docs/materials_features.md`, about 300 words. State which descriptors are mechanism-proximal (size, shape, charge, synthesis route) and which are compositional proxies, and say whether you expect dropping the Magpie block to help or hurt cross-paper generalization.

**This is a testable prediction.** If you say the Magpie block hurts, Henry or Matthew will test it. That makes it your result.

**Done when:** branch `edward/audit`, both docs committed, PR opened.

**Due Sunday August 10.**

---

# STEPHEN — Related work

**Why you, honestly.** Your interests are cardiovascular and molecular biology, and this project does not reach them. I am not going to pretend otherwise. What you have instead is the section that determines whether our framing survives contact with a reviewer, and you have already changed the direction of this project twice by finding Li and Xu.

**You own:** where we sit relative to published work.

### The search is over

Three passes found three papers, each closer than the last. That pattern does not converge. We are stopping and building.

### Step 1. Write it up (Friday to Sunday)

`docs/literature.md`. For each of Li, Xu, and Austin: full citation with journal, volume, pages, and DOI, then two or three sentences on what they did and what they left open.

Then a closing paragraph that synthesizes. Something along these lines, in your own words:

> The problem is documented (Li), batch correction on literature-extracted data is established (Xu), and unlabeled target-study adaptation is demonstrated (Austin). What remains untested is whether unlabeled adaptation to an unseen source publication can recover predictive accuracy on a literature-extracted dataset.

That closing paragraph is the part a reviewer actually reads.

**Two corrections to the draft you posted:**

- On Li, add that their recommendation was prospective, that the field should collect better-standardized data going forward. That sharpens the contrast: their fix is for future datasets, ours is for the one that exists.
- On DEBIAS-M's "online" setting, point me at the specific passage in the paper. If that came from a summary rather than the paper itself, verify it before it goes in.

### Step 2. One last targeted check (Saturday)

Only this question, then stop: has anyone applied **unsupervised domain adaptation** to improve prediction on a **completely unseen source study** in a **literature-extracted** dataset, in any field?

Terms: unsupervised domain adaptation, test-time adaptation, transductive learning, leave-one-study-out, crossed with literature-curated, literature-extracted, cross-study, inter-laboratory.

If you find one, tell Henry immediately. If not, we build.

**Done when:** branch `stephen/lit`, `docs/literature.md` committed, PR opened, every DOI clicked and confirmed to resolve.

**Due Sunday August 10.**

---

# HENRY

Proposal, applicability domain, calibration, integration, PR review, and the URTC submission deadline, which the chairs have still not confirmed.

---

## Check-in: Sunday August 10, 9pm

Post your numbers. Open your PR. If something is not done, say so Sunday rather than Wednesday.
