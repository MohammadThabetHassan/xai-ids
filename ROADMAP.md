# XAI-IDS Enhancement Roadmap

**Version:** 3.0.2  
**Last Updated:** April 2026

---

## Overview

This roadmap outlines the planned enhancements for the XAI-IDS project, organized by phase with estimated impact and effort levels.

---

## Phase 1: Quick Wins (Immediate)

**Goal:** Fix documentation and visibility issues that could mislead users about model capabilities.

| # | Task | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 1.1 | Add per-class metrics table to README | HIGH | 30 min | Complete |
| 1.2 | Add macro/weighted F1 prominently | HIGH | 30 min | Complete |
| 1.3 | Document class imbalance in Limitations | HIGH | 15 min | Complete |
| 1.4 | Create this roadmap document | HIGH | 1 hour | Complete |
| 1.5 | Create PROJECT_STATUS.md | HIGH | 1 hour | Complete |
| 1.6 | Add confusion matrix interpretation guide | MEDIUM | 1 hour | Complete |
| 1.7 | Update site with class performance visualization | MEDIUM | 2 hours | Complete |

---

## Phase 2: Research Quality Improvements (Month 1)

**Goal:** Improve scientific validity and evaluation rigor.

| # | Task | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 2.1 | Implement class-weighted training | HIGH | 2 hours | Complete |
| 2.2 | Add cross-validation (5-fold) | HIGH | 3 hours | Complete |
| 2.3 | Generate per-class precision-recall curves | HIGH | 2 hours | Complete |
| 2.4 | Add calibration analysis plots | HIGH | 2 hours | Complete |
| 2.5 | Generate failure analysis report | HIGH | 2 hours | Complete |
| 2.6 | Add statistical significance (paired t-test) | MEDIUM | 2 hours | Pending |
| 2.7 | Add confusion matrix per-class interpretation | MEDIUM | 1 hour | Complete |

---

## Phase 3: Multi-Dataset Evaluation (Complete)

**Goal:** Evaluate on multiple IDS datasets to establish generalizability.

| # | Task | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 3.1 | Find alternative CIC-IDS-2017 source | CRITICAL | 2 hours | Complete |
| 3.2 | Download/verify dataset integrity | CRITICAL | 1 hour | Complete |
| 3.3 | Run full benchmark on real data (~100K samples) | CRITICAL | 4 hours | Complete |
| 3.4 | Compare synthetic vs real performance | HIGH | 2 hours | Complete |
| 3.5 | Document dataset provenance | MEDIUM | 1 hour | Complete |
| 3.6 | Update all claims with real data results | HIGH | 2 hours | Complete |
| 3.7 | Add UNSW-NB15 dataset evaluation | HIGH | 4 hours | Complete |
| 3.8 | Add CSE-CIC-IDS-2018 dataset evaluation | HIGH | 4 hours | Complete |
| 3.9 | Add LightGBM model | HIGH | 2 hours | Complete |
| 3.10 | Add VotingEnsemble model | HIGH | 2 hours | Complete |
| 3.11 | Implement XCS (XAI Confidence Score) | HIGH | 3 hours | Complete |
| 3.12 | Cross-dataset feature importance analysis | MEDIUM | 3 hours | Complete |

---

## Phase 4: Engineering Hardening (Complete)

**Goal:** Production-ready infrastructure and testing.

| # | Task | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 4.1 | Unit tests for evaluation, explainability, and XCS | MEDIUM | 4 hours | Complete |
| 4.2 | Add regression tests for outputs | MEDIUM | 2 hours | Pending |
| 4.3 | Create Dockerfile | MEDIUM | 2 hours | Complete |
| 4.4 | Add FastAPI inference endpoint | MEDIUM | 3 hours | Complete |
| 4.5 | Replace simulated demo with real model API | HIGH | 3 hours | Complete |
| 4.6 | Add input validation and error handling | MEDIUM | 2 hours | Complete |
| 4.7 | Create configuration file (YAML/JSON) | LOW | 1 hour | Complete |

---

## Phase 5: Portfolio Polish (Complete)

**Goal:** Professional presentation and community engagement.

| # | Task | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 5.1 | Add GitHub issue templates | LOW | 30 min | Complete |
| 5.2 | Create CHANGELOG.md | LOW | 30 min | Complete |
| 5.3 | Add architecture diagram | MEDIUM | 1 hour | Complete |
| 5.4 | Create model card (like HuggingFace) | MEDIUM | 1 hour | Complete |
| 5.5 | Add architecture documentation (MkDocs) | MEDIUM | 2 hours | Complete |
| 5.6 | Create citation file (CITATION.cff) | LOW | 15 min | Complete |
| 5.7 | Create RESULTS.md with full tables | HIGH | 2 hours | Complete |
| 5.8 | Add CONTRIBUTING.md | LOW | 30 min | Complete |
| 5.9 | Add SECURITY.md | LOW | 30 min | Complete |
| 5.10 | Add PR template | LOW | 15 min | Complete |

---

## Dependency Graph

```
Phase 1 (Quick Wins)
    ↓
Phase 2 (Research Quality) ← depends on 1.1, 1.2
    ↓
Phase 3 (Multi-Dataset) ← depends on 2.1, 2.2
    ↓
Phase 4 (Engineering) ← can run in parallel after Phase 2
    ↓
Phase 5 (Portfolio) ← final polish after all above
```

---

## Success Metrics

| Phase | Metric | Target | Outcome |
|-------|--------|--------|---------|
| 1 | Per-class metrics visibility | All 15 classes shown in docs | Met |
| 2 | Macro F1 (synthetic split) | > 0.60 | Not met — 0.51 |
| 2 | Attack-class detection | All classes above 0% recall | Partial — improved, some classes still 0% |
| 3 | Real-data benchmark | 3 datasets evaluated | Met — CIC-IDS-2017, UNSW-NB15, CSE-CIC-IDS-2018 |
| 3 | Best XGBoost accuracy | > 99% | Met — 99.66% on CIC-IDS-2017 |
| 4 | Test coverage | > 80% line coverage | Not measured — 24 tests, coverage not yet gated |

---

## Notes

- This roadmap is a living document and will be updated as priorities shift
- Effort estimates assume single developer working 2-4 hours/day
- Multi-dataset evaluation completed via Kaggle notebook (Tesla T4 GPU)
- XCS (XAI Confidence Score) implemented: XCS = 0.4×Confidence + 0.3×(1-SHAP_Instability) + 0.3×Jaccard(SHAP,LIME)

---

*Last updated: April 2026*
