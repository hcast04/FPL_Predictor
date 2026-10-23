"""Constants and paths. Nothing here depends on anything else in the package."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # the FPL project root
DATA = ROOT / "data"
HISTORICAL = DATA / "historical_fpl.csv.gz"
ARCHIVE = DATA / "online_forecast_archive_v1.json.gz"
FOLD_CACHE = DATA / "walk_forward_fold_cache_base_v2_4_0"

# --- FPL rules. These are the game, not choices. ---------------------------
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_CLUB = 3
BUDGET = 100.0            # GBP millions
POSITIONS = ("GK", "DEF", "MID", "FWD")
HIT_COST = 4              # points per transfer beyond the free allowance
MAX_BANKED_TRANSFERS = 5

# --- evaluation --------------------------------------------------------------
HOLDOUT_SEASONS = ("2023-24", "2024-25", "2025-26")
MIN_PLAYERS_PER_GAMEWEEK = 50   # below this a gameweek is unusable

# The join key. web_name is NOT unique ("Palmer" is two different players).
PLAYER_KEY = "player_code"
