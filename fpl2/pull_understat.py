#!/usr/bin/env python3
"""
Re-scrape Understat per-match player stats for the live season.

`no_cache=True` forces a fresh download. A cached read would silently
reproduce whatever gameweek the cache stopped at, which is the whole failure
this script exists to prevent.

WHEN IT FAILS. Understat is a third-party site with no API contract. On
2026-11-05 it went down for maintenance -- every URL returned one
26,719-byte page reading "We will come back soon" -- and `soccerdata` raised
`JSONDecodeError: Expecting value: line 1 column 1` from deep inside
`_read_leagues`, which reads as a parser bug rather than an outage. This
script checks the site FIRST and says which of the two it is, because the
remedy is completely different: wait, versus fix the scraper.

The existing pickle is never overwritten unless the new scrape is at least as
large, so a partial read cannot destroy a good file.
"""
from __future__ import annotations
import re, sys, time, urllib.request
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C          # noqa: E402  pins SOCCERDATA_DIR

SEASON, CODE = "2026-27", "2627"
OUT = C.ROOT / "data" / "raw_extra" / f"understat_{SEASON}.pkl"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def site_is_up() -> tuple[bool, str]:
    """Is understat.com serving data, or a holding page?"""
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            "https://understat.com/league/EPL/2026", headers={"User-Agent": UA}),
            timeout=30)
        html = r.read().decode("utf8", "ignore")
    except Exception as e:                                    # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    if "JSON.parse" in html:
        return True, "page carries embedded data"
    return False, f"holding page ({len(html)}B): {text[:80]!r}"


def main() -> int:
    up, why = site_is_up()
    print(f"understat.com: {'UP' if up else 'DOWN'} -- {why}", flush=True)
    if not up:
        print("\nNOT a scraper bug: the site is not serving data. Wait and retry.\n"
              "The existing pickle is left untouched; `understat.staleness()`\n"
              "reports how far behind it now is.", flush=True)
        return 2

    import soccerdata as sd
    t0 = time.time()
    us = sd.Understat(leagues=["ENG-Premier League"], seasons=[CODE], no_cache=True)
    d = us.read_player_match_stats()
    r = d.reset_index()
    r["date"] = pd.to_datetime(r["game"].str.slice(0, 10))
    print(f"[{time.time()-t0:.0f}s] scraped {d.shape}: {r.game.nunique()} games, "
          f"{r.date.min().date()} -> {r.date.max().date()}", flush=True)

    if OUT.exists():
        prev = pd.read_pickle(OUT)
        if len(d) < len(prev):
            print(f"REFUSING to overwrite: new {len(d):,} rows < existing "
                  f"{len(prev):,}. A partial scrape must not replace a good file.")
            return 3
        print(f"  was {len(prev):,} rows -> now {len(d):,}")
    d.to_pickle(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
