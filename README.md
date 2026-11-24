# FarmLink Buyer Analytics

Retention, churn and causal analysis for a B2B agricultural marketplace. The
question it's built around is one a product roadmap actually turns on: when we
short-ship a buyer, how much does that really cost us in churn, and is it worth
building supply guarantees to stop it?

Built during a Data Analyst internship at FarmLink, an IIT Madras-incubated
Indian agri supply-chain startup. My scope was the analysis pipeline and the
models; the team runs them on FarmLink's internal data.

FarmLink's buyer ledger is confidential and no public dataset describes B2B
agricultural purchasing, so the order book here is simulated. It's driven by
real market data though: demand, fill rates and churn all respond to a genuine
2016-2026 daily wholesale vegetable price series (Central Bank of Sri Lanka
daily reports, the same public dataset used in the companion pricing repo), so
the ledger inherits real seasonality, real volatility clustering and a real
supply crisis.

---

## The headline finding

The naive answer is wrong by 7x, and it's wrong in the direction that would have
caused a bad roadmap decision.

| Method | Estimated effect on 8-week churn | 95% CI | Error vs truth |
|---|---:|---|---:|
| Naive difference in means | +18.51 pp | [16.24, 20.78] | +15.9 pp ✗ |
| Regression adjustment | +4.01 pp | [3.91, 4.11] | +1.4 pp ✗ |
| Propensity-score matching (5-NN, logit caliper) | +3.26 pp | [0.78, 5.74] | +0.7 pp ✓ |
| IPW (ATT-weighted, trimmed) | +4.10 pp | [0.69, 7.51] | +1.5 pp ✓ |
| AIPW (doubly robust) | +2.83 pp | [−0.46, 6.12] | +0.24 pp ✓ |
| Ground truth | +2.59 pp | — | — |

A short-shipped order raises a buyer's 8-week churn probability by about 2.6
percentage points, not the 18.5 the raw comparison suggests.

The reason for the gap is that short-shipping happens during supply shortages,
and shortages coincide with price spikes that drive buyers away on their own.
On top of that, small buyers get de-prioritised on allocation *and* churn more
for unrelated commercial reasons. Both channels inflate the raw difference.

### Why these numbers can be trusted

The order book comes from a written-down data-generating process, so the true
causal effect is known exactly, and the estimators never see it. That turns
"here is a causal estimate" into "here is a causal estimate from a method shown
to recover the right answer under this exact confounding structure", which is a
much more useful claim.

Covariate balance confirms the adjustment is doing real work:

| Covariate | \|SMD\| before | \|SMD\| after |
|---|---:|---:|
| prior fill-rate mean | 1.594 | 0.086 |
| basket volatility | 1.360 | 0.007 |
| prior short-ship rate | 1.070 | 0.065 |
| buyer size (log) | 0.856 | 0.060 |
| price shock | 0.824 | 0.013 |

All ten covariates come in under the conventional |SMD| < 0.1 bar, down from a
maximum of 1.59. Estimation is restricted to the region of common propensity
support, and the trimmed share is reported rather than quietly dropped.

---

## Everything else

### Cohort retention (39 monthly cohorts)

| Month | 1 | 3 | 6 | 12 |
|---|---:|---:|---:|---:|
| Retention | 82.5% | 66.7% | 51.5% | 33.1% |

Reported by cohort rather than blended. A growing marketplace shows a flattering
aggregate retention number even while every individual cohort is getting worse,
because new buyers keep refilling the top of the funnel.

### Churn model (out-of-time)

Trained on a temporal split. Buyer-weeks are serially correlated, so a random
split leaks a buyer's own future into its training rows and gives you an AUC that
disappears in production.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.742 |
| PR-AUC | 0.239 (3.50x base rate) |
| Top-decile precision | 24.5% (3.59x lift) |
| Brier score | 0.059 |
| Base churn rate (8-week) | 6.8% |

Top drivers by mean \|SHAP\|: buyer size (25.9%), basket volatility (10.0%),
hotel segment (9.7%), quality preference (9.7%), prior fill-rate mean (8.7%).

### The threshold comes from money, not F1

The question isn't whether the classifier is good, it's what score justifies
spending LKR 2,500 on a retention call. With the margin at risk derived from the
data (26 weeks of forward contribution) and an assumed 30% save rate:

| Threshold | Contacted | Precision | Recall | Net value |
|---:|---:|---:|---:|---:|
| 0.09 | 3,681 | 16.6% | 61.9% | +13.1 M LKR |

### Experiment design for the intervention this implies

Detecting a 20% relative reduction in churn needs 4,329 buyers per arm, about
9.6 weeks at current scale. CUPED on pre-period GMV cuts variance 42.2%
(rho = 0.65), worth a 1.73x effective sample size, which turns a 10-week
experiment into roughly a 6-week one. O'Brien-Fleming boundaries and an SRM
check are in there because peeking at a running experiment at a fixed alpha of
0.05 pushes the false-positive rate to about 14% over five looks.

---

## How the simulation works

Confounding is built in on purpose, mirroring the real mechanism. Large buyers
get prioritised during shortages, so they see better fill, and they also churn
less commercially. Price-sensitive segments buy volatile crops, so they see worse
fill, and they're independently likelier to leave.

Churn is a weekly hazard driven by an exponentially weighted memory of bad fills,
plus buyer size, tenure, inactivity and real price shocks. Both potential
outcomes get computed analytically by re-propagating that memory with the
exposure forced on and off while holding all other randomness fixed, so the
ground-truth ATT has no Monte Carlo noise in it.

Scale: 900 buyers, 129,207 order lines, 34,739 buyer-weeks, 166 weeks
(Jan 2023 to Mar 2026), 1.57 B LKR GMV, mean fill rate 94.9%, 13.2% of orders
short-shipped.

---

## Layout

```
src/farmlink_buyers/
  prices.py      real CRAN/CBSL price series -> weekly scarcity + volatility signals
  simulate.py    order-book generator; records potential outcomes for validation
  metrics.py     cohort retention, RFM, k-means personas, LTV, north-star tree
  churn.py       temporal split, XGBoost + isotonic calibration, SHAP, EV threshold
  causal.py      propensity scores, common support, PSM, IPW, AIPW, balance diagnostics
  experiment.py  power/MDE, CUPED, sequential boundaries, SRM check
scripts/run_analysis.py   reproduces every number above
tests/test_pipeline.py    22 tests
```

## Running it

```bash
make setup && make run && make test
```

Outputs land in `reports/results/` as CSV, plus `headline.json`.

---

## Limitations

The order book is synthetic. The methodology transfers, but the specific
retention and LTV figures are properties of the simulation and not of FarmLink.
What's validated here is the causal method, not the causal number.

The ground-truth validation is partly circular by construction, since the
estimators are checked against a process whose confounders they were handed. The
part that isn't circular is that the naive estimate still fails badly on the same
data.

Unobserved confounding isn't simulated. Real buyers churn for reasons no ledger
records: a chef leaves, a competitor undercuts. A sensitivity analysis bounding
that is the next step.

The 30% save rate in the threshold model is assumed rather than measured, which
is exactly what the proposed experiment would establish.

## Licence

Code is MIT. The price dataset is redistributed from CRAN under GPL-3 and is
fetched at runtime rather than vendored here.
