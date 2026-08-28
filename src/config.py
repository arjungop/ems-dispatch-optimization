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
