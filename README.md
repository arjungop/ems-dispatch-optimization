# Dynamic EMS Dispatch & Station Placement Optimization

Optimizing where ambulances wait and how they are dispatched, using real Seattle 911 data.

**Operations Research · Project Review 1** — Aadith Krishna (CB.SC.U4AIE23202),
Aditya Santosh (CB.SC.U4AIE23207), Suryansh Ram Menon (CB.SC.U4AIE23255),
Arjun Gopal (CB.SC.U4AIE23271).

In a medical emergency, survival is decided in minutes. Two operational decisions control
how fast an ambulance actually reaches a patient, and both can be optimized:

- **Q1 — Where to wait?** Placing stations well means more of the city sits within a short
  response-time radius.
- **Q2 — Who to send?** During simultaneous emergencies, smart ambulance-to-incident
  matching cuts everyone's wait.

## What this is

A Streamlit application that loads **2.2 million real Seattle 911 records** and runs three
operations-research models over them:

| | Model | Question it answers |
|---|---|---|
| **AHP** | Multi-criteria scoring | Which areas most deserve a station, weighing incident density, distance from existing help, and coverage potential? |
| **Covering** | Integer program (ILP) | Where should stations go so every zone is reachable within an *X*-minute limit — or, on a fixed budget of *P* stations, so the most demand is covered? |
| **Assignment** | Linear program (LP) | In a peak hour of simultaneous emergencies, which ambulance goes to which incident to minimise total response time? |

The two optimisations are linked: **the stations chosen by the covering model become the
ambulance bases used by the assignment model.**

Headline results over the last 12 months (79,152 EMS incidents, 28 demand zones):

- **7 stations** give 100% coverage at an 8-minute limit — against Seattle's 36 real stations,
  because placement follows demand rather than history.
- The assignment LP cuts total fleet travel time **8.2%** versus nearest-unit-first greedy
  dispatch on the busiest hour in the period, measured over the same calls on both sides.
- Every model is verified on the real data, including a cross-check of the LP against an
  independent Hungarian-algorithm solution.

---

## Quick start

> **The datasets are not in this repository.** The raw 911 feed is ~191 MB, over GitHub's
> file-size limit, so `data/` is generated locally by the first two commands below.

```bash
pip install -r requirements.txt

python scripts/download_data.py    # ~2.2M real 911 records + stations + ZIP polygons
python scripts/preprocess.py       # clean, EMS-filter, tag ZIP zones, cache to Parquet
streamlit run app.py
```

### Solvers

The models are written with PuLP, which bundles a CBC binary built for x86_64
only — on Apple Silicon without Rosetta every solve fails with *"Bad CPU type in
executable"*. `requirements.txt` therefore also installs **HiGHS**, a native
wheel that works on every current platform. `config.get_solver()` probes the
available solvers once, by actually solving a trivial LP, and uses the first
that runs: HiGHS, then a system CBC on `PATH` (`brew install cbc`), then PuLP's
bundled copy. Nothing needs configuring — but if you ever want to know which one
is in play, `config.solver_name()` will tell you.

The download pages the full dataset from the open-data portal and takes roughly
20–25 minutes; preprocessing takes a couple of minutes. Both are one-time costs —
after that the app starts in seconds. To try the pipeline on a smaller slice first,
`python scripts/download_data.py --limit 200000`.

Everything is downloaded from public, open-licence endpoints — no API keys or accounts
are needed.

---

## The data

| Source | What it is | Used for |
|---|---|---|
| **Seattle Real-Time Fire 911 Calls** — `data.seattle.gov`, dataset ID `kzjm-xkqj` | ~2.2M incident records, updated every 5 minutes, public domain | The **demand** side: where and when real emergencies occur |
| **Seattle Fire Stations** — Seattle City GIS | The 36 stations Seattle actually operates | Benchmarking our optimized placement against reality |
| **ZIP Code Tabulation Areas** — US Census TIGERweb | ZIP polygons covering the city | Computing each incident's zone |

Fields used from the 911 feed:

| Field | Meaning | Used for |
|---|---|---|
| `address` / `latitude` / `longitude` | Exact incident location | Mapping demand, distances |
| `type` | Kind of call (medical, aid, fire…) | Filtering to EMS calls only |
| `datetime` | Date and time of the call | Peak-hour stats, dispatch scenario |
| `zip code` (computed) | Which zone the incident is in | Grouping demand into zones |
| `incident_number` | Unique ID per call | De-duplication |

---

## Methodology

```
Data in → clean & compute stats → AHP ranks sites → two optimizations run → results on the map
```

### Step 1 — Data cleaning & statistics (`src/data_loader.py`, `src/stats.py`)

The raw feed mixes fire suppression, alarms, hazmat and EMS in one `type` column, so
calls are filtered to genuine EMS/aid dispatches using whole-word patterns (`aid`,
`medic`, `medical`, `AMR`, `nurseline`, `triaged`, `low acuity`, `overdose`, …). A
second, opt-in tier adds incidents that generate patients but are typed by their cause
(collisions, rescues, violence). Rows are de-duplicated on incident number, bounded to
Seattle, and tagged with a ZIP zone by point-in-polygon join.

The feed is harvested live, so its final calendar day is cut off partway through.
That day is flagged rather than dropped: the daily-trend chart draws it dashed and
labels it *final day partial*, so the short bar reads as a truncated harvest rather
than a collapse in demand.

Each ZIP zone becomes a demand point whose weight is its incident count, and whose
location is the mean incident coordinate — putting the candidate site where the calls
actually are rather than at the polygon's geometric centre.

### Step 2 — AHP multi-criteria site scoring (`src/ahp.py`)

Three competing criteria are combined into one priority score per candidate zone:

| Criterion | Meaning |
|---|---|
| **Incident Density** | Number of emergencies in the area |
| **Equity Gap** | Distance to the nearest existing station (farther = higher need) |
| **Coverage Potential** | How many zones a station here could reach quickly |

Criteria are compared pairwise on the **Saaty 1–9 scale** (3 comparisons for 3 criteria).
Weights are the principal eigenvector of the comparison matrix. Because pairwise ratings
are human judgements that can quietly contradict each other, AHP measures this:

**λmax → Consistency Index CI = (λmax − n)/(n − 1) → Consistency Ratio CR = CI/RI**

**CR < 0.1** means the judgements are consistent and the weights are trustworthy; above
that, the comparisons are revised. Each member fills the table independently and the
matrices are combined with the element-wise **geometric mean** — the only average that
keeps the matrix reciprocal, so if members' views of A-vs-B average to 3, their view of
B-vs-A automatically averages to 1/3.

The four members' own ratings are recorded in `TEAM_JUDGEMENTS` in
`src/config.py` — the single place they live, so the app, the chart renderer and
the verification suite cannot drift apart. Every member's matrix is internally
consistent (**CR = 0.000** for all four, and for the aggregate), which is a
property of the ratings rather than luck: each member's density-vs-coverage
rating is exactly the product of their other two, so the judgements contain no
contradiction for the consistency check to find.

The combined weights are **Incident Density 67.4%, Equity Gap 23.1%, Coverage
Potential 9.4%** — the team weights raw call volume well above the other two,
and rates the ability to blanket several zones from one site as the least
important of the three.

### Step 3 — Optimization 1: Covering model, ILP (`src/covering.py`)

A zone is *covered* if a station can reach it within the response-time limit.

**Mode A — Set Covering.** Minimum number of stations such that every demand zone is
served within the limit.

```
minimise    Σ y_i
subject to  Σ_{i ∈ N(j)} y_i ≥ 1     for every zone j
            y_i ∈ {0,1}
```

**Mode B — Maximal Covering.** With the station count fixed at P, serve the greatest
possible share of demand — the budget-constrained reality where full coverage may be
infeasible.

```
maximise    Σ d_j z_j
subject to  z_j ≤ Σ_{i ∈ N(j)} y_i    for every zone j
            Σ y_i = P
            y_i, z_j ∈ {0,1}
```

The app also sweeps the limit at **5 / 8 / 10 minutes** and benchmarks the optimized
placement against Seattle's 36 real stations under identical assumptions.

### Step 4 — Optimization 2: Assignment model, LP (`src/assignment.py`)

During a peak hour of simultaneous emergencies, which ambulance goes to which incident?

```
minimise    Σ_ij c_ij x_ij
subject to  Σ_i x_ij = 1      each incident → exactly one ambulance
            Σ_j x_ij ≤ 1      each ambulance → at most one incident
            x_ij ≥ 0
```

The constraint matrix is totally unimodular, so the LP relaxation is guaranteed to have
an integral optimum — which is why this is an LP rather than an ILP. Unequal counts are
handled the transportation-problem way, by padding the cost matrix with zero-cost dummy
rows or columns so the problem is balanced and always feasible.

**The connecting idea:** the station locations determined by the covering model serve as
the ambulance bases in the assignment model. This linkage integrates both optimizations
into a single, coherent framework — the core contribution of the project.

The app also reports how much the optimal plan beats nearest-unit-first greedy
dispatch. Because a peak hour normally carries more calls than ready units, and
an unserved call is free in the objective, that comparison is deliberately
restricted to the same workload on both sides — the first *k* calls by arrival,
*k* being the fleet size — so the gap measures matching quality rather than
which calls each method chose to answer (`benchmark_vs_greedy` in
`src/assignment.py`).

### Travel-time model (`src/geo.py`)

Haversine gives great-circle distance; real ambulances follow streets. So straight-line
distance is inflated by a **circuity factor** (default 1.30), converted at an **effective
response speed** (default 45 km/h), and a **turnout time** (default 1 min) is added, so a
threshold means *time from alarm to on-scene*. All three are adjustable in the sidebar,
and every model re-solves against the current settings.

---

## Project layout

```
app.py                     Streamlit application (6 tabs)
requirements.txt
.streamlit/config.toml     Pinned light theme, so the app looks the same anywhere
scripts/
  download_data.py         Fetch all three real datasets
  preprocess.py            Clean → EMS-filter → ZIP-tag → Parquet cache
  verify_pipeline.py       End-to-end checks of every model on the real data
  test_app.py              Headless smoke test of the Streamlit app
  render_charts.py         Render every chart to one static HTML page
src/
  config.py                Paths, EMS taxonomy, travel-time & AHP constants
  geo.py                   Haversine, travel time, point-in-polygon ZIP join
  data_loader.py           Step 1a — ingestion, cleaning, zones, peak hours
  stats.py                 Step 1b — on-demand statistics
  ahp.py                   Step 2 — pairwise comparison, weights, CR check
  covering.py              Step 3 — set covering & maximal covering ILP
  assignment.py            Step 4 — dispatch assignment LP
  viz.py                   Plotly charts & Folium maps
data/
  raw/                     Downloaded source data
  processed/               Parquet cache
```

## Verification

```bash
python scripts/verify_pipeline.py   # 59 checks — every model, on the real data
python scripts/test_app.py          # 37 checks — drives the app headlessly
```

`test_app.py` also checks the presentation layer — that the methodology diagram
renders with both optimisations and the covering→assignment linkage, and that the
team roster and the per-member AHP tabs carry the right names, and that the four
project objectives, the project's significance and the six tools are all stated —
so the parts of the deliverable that are looked at rather than computed cannot
silently regress.

`verify_pipeline.py` asserts the properties each model must satisfy, not just that
it runs: set covering must reach full coverage of every reachable zone; maximal
covering must open exactly P and be non-decreasing in P; MCLP at P = the
set-cover optimum must reach the same coverage; a looser response limit must
never require more stations; AHP weights must sum to 1 with λmax ≥ n and CR <
0.1; the greedy benchmark must put the same workload on both sides and the LP
must never lose to it; and the assignment LP is cross-checked against an
independent Hungarian-algorithm solution of the same balanced matrix.

Beyond those properties, a final section proves optimality outright: both
covering models are re-solved on small synthetic cities and checked against an
exhaustive enumeration of every possible station subset, the AHP weights are
checked against the largest-modulus eigenvector on 200 random reciprocal
matrices, and the dispatch LP is checked against greedy on 150 random cost
matrices. A solver swap or a drift in how a model is stated therefore cannot
pass unnoticed. Both suites currently pass in full — 59 and 37 checks, 0
failures.

### Results on the real data

Over the last 12 months (79,152 EMS incidents, 28 demand zones, default travel
assumptions):

| Response limit | Stations needed | Demand covered | Seattle's 36 real stations |
|---|---|---|---|
| 5 min | 16 | 100% | 99.72% |
| 8 min | 7 | 100% | 100% |
| 10 min | 5 | 100% | 100% |

The optimised plan reaches comparable coverage from far fewer sites because it is
placed against demand rather than accumulated historically — and because the real
network also serves fire suppression, which this EMS-only model does not price in.
On the busiest hour in the period (Wed 19 Nov 2025, 09:00, 30 simultaneous
incidents) the assignment LP beat nearest-unit-first greedy dispatch on total
fleet travel time:

| Ready units | Fleet | Optimal | Greedy | LP better by |
|---|---|---|---|---|
| 1 per station | 7 units | 74.40 min | 81.00 min | **8.2%** |
| 2 per station | 14 units | 143.96 min | 149.91 min | **4.0%** |

Both sides are measured over the same calls — the first *k* to arrive, where *k*
is the fleet size. That restriction matters: a peak hour has more calls than
units, and unserved calls cost nothing in the objective, so an unrestricted LP
would quietly answer only the *nearest* calls while greedy took them as they
came. Comparing those totals would score triage rather than routing and inflate
the LP's margin several-fold (to 62% and 40% respectively). The figures above
are the honest routing gain.

### A note on fringe zones

The feed carries a long tail of ZIP zones holding a handful of mutual-aid calls a
year (Renton, SeaTac, Medina and similar), which sit outside the city. Set
covering treats a one-call zone exactly like a ten-thousand-call zone, so left in,
each would demand its own dedicated station — inflating the 8-minute answer from 7
stations to 13, with six of them serving 0.04% of demand. A **materiality floor**
(default 0.1% of period demand, adjustable in the sidebar) excludes them from the
models while keeping them visible in the tables. It retains 99.8% of all demand.

## Tools & tech stack

**Python + pandas** loading data & computing statistics · **PuLP + HiGHS** modelling and
solving the covering (ILP) & assignment (LP) models · **Streamlit** the interactive user interface ·
**Folium / Plotly** mapping incidents & stations · **Haversine** estimating travel
distance / time · **Data ingestion** export CSV from portal / API into the app.
