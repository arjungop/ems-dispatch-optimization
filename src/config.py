"""Central configuration: paths, EMS call taxonomy, and travel-time defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

CALLS_CSV = RAW / "seattle_911_calls.csv"
STATIONS_CSV = RAW / "seattle_fire_stations.csv"
ZCTA_GEOJSON = RAW / "seattle_zcta.geojson"
CALLS_PARQUET = PROCESSED / "ems_incidents.parquet"

# --------------------------------------------------------------------------
# EMS call taxonomy
# --------------------------------------------------------------------------
# The raw feed mixes fire suppression, alarms, hazmat and EMS in one `type`
# column. Two tiers let the user decide how wide "an EMS call" is:
#
#   CORE    -- unambiguous aid / medic dispatches. This is the default and is
#              what the project means by "genuine EMS/aid calls".
#   TRAUMA  -- incidents that generate patients but are typed by their cause
#              (collisions, rescues, violence). Opt-in from the sidebar.
#
# Patterns are matched as whole words so "aid" cannot match "raid"/"afraid".
EMS_CORE_PATTERNS = [
    r"\baid\b",
    r"\bmedic\b",
    r"\bmedics\b",
    r"\bmedical\b",
    r"\bems\b",
    r"\bamr\b",
    r"\bnurseline\b",
    r"\btriaged\b",
    r"\blow acuity\b",
    r"\boverdose\b",
    r"\bcardiac\b",
    r"\bcpr\b",
]

EMS_TRAUMA_PATTERNS = [
    r"\bmvi\b",
    r"\bmotor vehicle\b",
    r"\brescue\b",
    r"\bscenes of violence\b",
    r"\bassault\b",
    r"\bcasualty\b",
    r"\bcasualties\b",
    r"\bdrowning\b",
    r"\bextrication\b",
]

# --------------------------------------------------------------------------
# Travel-time model (Haversine straight-line -> road-network estimate)
# --------------------------------------------------------------------------
# Haversine gives great-circle distance; real ambulances follow streets, so the
# straight-line distance is inflated by a circuity factor before being divided
# by an effective emergency response speed. Turnout time (crew getting rolling)
# is added on top so the threshold means "time from alarm to on-scene".
DEFAULT_SPEED_KMH = 45.0
DEFAULT_CIRCUITY = 1.30
DEFAULT_TURNOUT_MIN = 1.0

# Response-time thresholds the project explicitly studies.
RESPONSE_THRESHOLDS_MIN = [5, 8, 10]

# --------------------------------------------------------------------------
# AHP
# --------------------------------------------------------------------------
AHP_CRITERIA = ["Incident Density", "Equity Gap", "Coverage Potential"]

AHP_CRITERIA_HELP = {
    "Incident Density": "Number of emergencies in the area.",
    "Equity Gap": "Distance to the nearest existing station (farther = higher need).",
    "Coverage Potential": "How many zones a station here could reach quickly.",
}

# Saaty's Random Consistency Index, indexed by matrix order n.
SAATY_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
            8: 1.41, 9: 1.45, 10: 1.49}

CR_THRESHOLD = 0.10

SEATTLE_CENTER = (47.6062, -122.3321)


# --------------------------------------------------------------------------
# LP / ILP solver selection
# --------------------------------------------------------------------------
# PuLP bundles a CBC binary, but that copy is built for x86_64 only. On Apple
# Silicon without Rosetta every solve dies with "Bad CPU type in executable",
# which takes both optimisations down while the data and AHP steps still pass.
# The team works on different laptops, so rather than pin one solver we probe
# for the first that genuinely runs here, in order of preference:
#
#   1. HiGHS       native wheel (`pip install highspy`), arm64 included
#   2. system CBC  e.g. `brew install cbc`, picked up from PATH
#   3. bundled CBC PuLP's own copy -- correct on Intel/Windows/Linux
#
# Availability is settled by actually solving a trivial LP, not by asking the
# solver whether it thinks it exists: PuLP reports the bundled CBC as available
# on arm64 because the file is present, and only exec fails.
_SOLVER_CHOICE: tuple[str, object] | None = None


def _solver_candidates():
    """Yield (name, factory) pairs in order of preference."""
    import shutil

    import pulp

    yield "HiGHS", lambda msg: pulp.HiGHS(msg=msg)

    cbc_path = shutil.which("cbc")
    if cbc_path:
        yield "CBC (system)", lambda msg: pulp.COIN_CMD(path=cbc_path, msg=msg)

    yield "CBC (bundled)", lambda msg: pulp.PULP_CBC_CMD(msg=msg)


def _solver_works(solver) -> bool:
    """Solve a one-variable LP to prove the solver really runs on this machine."""
    import pulp

    probe = pulp.LpProblem("solver_probe", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    probe += x
    probe += x >= 1
    try:
        status = probe.solve(solver)
    except Exception:
        return False
    return pulp.LpStatus.get(status) == "Optimal"


def get_solver(msg: bool = False):
    """Return a PuLP solver that is known to run here, probing once and caching."""
    global _SOLVER_CHOICE

    if _SOLVER_CHOICE is None:
        for name, factory in _solver_candidates():
            try:
                if _solver_works(factory(False)):
                    _SOLVER_CHOICE = (name, factory)
                    break
            except Exception:
                continue
        else:
            raise RuntimeError(
                "No working LP/ILP solver found. Install one with "
                "`pip install highspy` (recommended) or `brew install cbc`."
            )

    return _SOLVER_CHOICE[1](msg)


def solver_name() -> str:
    """Name of the solver in use, for display in the app and the check suites."""
    get_solver()
    return _SOLVER_CHOICE[0]


# --------------------------------------------------------------------------
# Project identity — the title slide of the review deck
# --------------------------------------------------------------------------
PROJECT_TITLE = "Dynamic EMS Dispatch & Station Placement Optimization"
PROJECT_SUBTITLE = (
    "Optimizing where ambulances wait and how they are dispatched, "
    "using real Seattle 911 data"
)
COURSE = "Operations Research · Project Review 1"

# Ordered as they appear on the title slide. Each member fills the AHP
# comparison table independently, so this roster also names the AHP tabs.
TEAM = [
    ("Aadith Krishna", "CB.SC.U4AIE23202"),
    ("Aditya Santosh", "CB.SC.U4AIE23207"),
    ("Suryansh Ram Menon", "CB.SC.U4AIE23255"),
    ("Arjun Gopal", "CB.SC.U4AIE23271"),
]


# --------------------------------------------------------------------------
# AHP pairwise judgements, one set per team member
# --------------------------------------------------------------------------
# Each member compares the three criteria two at a time on the Saaty 1-9 scale.
# Index order follows AHP_CRITERIA: 0 = Incident Density, 1 = Equity Gap,
# 2 = Coverage Potential. So (0, 1): 3.0 reads "Incident Density is judged three
# times as important as Equity Gap"; a value below 1 means the reverse.
#
# This is the one place these numbers live. The app seeds each member's tab from
# here, verify_pipeline.py checks them, and render_charts.py draws them -- so the
# team's judgements stay consistent everywhere instead of drifting between an
# unsaved slider, a test fixture and a chart.
#
# JUDGEMENTS_ARE_REAL records whether these are the team's own ratings or the
# stand-in that lets the app open with something coherent. The app shows a
# warning while it is False, because the project asks each member to supply
# these personally and presenting a stand-in as their input would misrepresent
# the one part of the methodology that is theirs. Flip it to True in the same
# edit that replaces the numbers below.
JUDGEMENTS_ARE_REAL = True

TEAM_JUDGEMENTS: dict[str, dict[tuple[int, int], float]] = {
    # (0, 1) Incident Density vs Equity Gap
    # (0, 2) Incident Density vs Coverage Potential
    # (1, 2) Equity Gap     vs Coverage Potential
    "Aadith Krishna":     {(0, 1): 3.0, (0, 2): 6.0, (1, 2): 2.0},
    "Aditya Santosh":     {(0, 1): 2.0, (0, 2): 6.0, (1, 2): 3.0},
    "Suryansh Ram Menon": {(0, 1): 4.0, (0, 2): 8.0, (1, 2): 2.0},
    "Arjun Gopal":        {(0, 1): 3.0, (0, 2): 9.0, (1, 2): 3.0},
}


def judgements_for(member_name: str) -> dict[tuple[int, int], float]:
    """One member's pairwise ratings, falling back to the first set on record."""
    if member_name in TEAM_JUDGEMENTS:
        return TEAM_JUDGEMENTS[member_name]
    return next(iter(TEAM_JUDGEMENTS.values()))
