#!/usr/bin/env python3
"""
Scrape every Premier League managerial spell into `data/raw_extra/managers.csv`.

WHY. The model has no manager feature at all, and the FPL API has no manager
field -- `teams` carries form and strength ratings, `element_types` is the
four playing positions. A new appointment reaches the model ONLY through
rolling outcome windows (`team_scored_r5`, `opp_def_r10`), which lag five to
ten gameweeks. So a bounce or a collapse in the weeks after a change is
invisible, then arrives slowly and already half-diluted. O60 has sat in the
ledger unstarted precisely because nobody had appointment dates.

SOURCE. Wikipedia's "List of Premier League managers", which carries one row
per spell with club, start and end. It is the only source found that covers
all eleven seasons in this project AND the live one: on 2026-11-24 it already
had Pochettino at Manchester United from 18 November 2026, six days old.

CARETAKERS ARE KEPT AND FLAGGED. The page marks them with a dagger; a
caretaker's three games are a different thing from an appointment and the
distinction must survive into the data rather than be decided here.

This writes RAW rows. Mapping clubs onto the panel's teams is
`fpl2/fpl2/managers.py`, because that is where the join can be tested.
"""
from __future__ import annotations
import io, re, sys, unicodedata
from pathlib import Path
import urllib.request
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C                                   # noqa: E402

URL = "https://en.wikipedia.org/wiki/List_of_Premier_League_managers"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
OUT = C.ROOT / "data" / "raw_extra" / "managers.csv"
# the table carries a dagger for a sitting manager and a double dagger for a
# caretaker; both must come off the name and one of them is information
CARETAKER, INCUMBENT = "‡", "†"


def _clean(name: str) -> tuple[str, bool, bool]:
    s = str(name)
    care, inc = CARETAKER in s, INCUMBENT in s
    s = s.replace(CARETAKER, "").replace(INCUMBENT, "").strip()
    s = unicodedata.normalize("NFC", re.sub(r"\[\d+\]", "", s)).strip()
    return s, care, inc


def fetch(url: str = URL) -> pd.DataFrame:
    html = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": UA}),
        timeout=60).read().decode("utf8", "ignore")
    tabs = pd.read_html(io.StringIO(html))
    # the spells table is the one carrying From/Until, not the summary tables
    cand = [t for t in tabs if {"Name", "Club", "From", "Until"} <= set(t.columns)]
    if not cand:
        raise RuntimeError("no table with Name/Club/From/Until -- page changed")
    t = max(cand, key=len).copy()
    nm = t["Name"].map(_clean)
    t["manager"] = [a for a, _, _ in nm]
    t["caretaker"] = [b for _, b, _ in nm]
    t["incumbent"] = [c for _, _, c in nm]
    t["club"] = t["Club"].map(lambda s: re.sub(r"\[\d+\]", "", str(s)).strip())
    # Wikipedia footnotes a date as "30 August 2022[a]" -- LETTER markers, not
    # the numeric ones stripped from names. Eleven spells failed to parse
    # before this, among them Solskjaer at Manchester United and De Zerbi at
    # Brighton, and `errors="coerce"` dropped them silently.
    def _date(col):
        return pd.to_datetime(
            t[col].astype(str).str.replace(r"\[[a-z0-9]+\]", "", regex=True)
                  .str.replace(r"Present\*?", "", regex=True).str.strip(),
            errors="coerce", format="mixed", dayfirst=True)

    t["start"] = _date("From")
    # "Present*" for a sitting manager leaves NaT, which is not missing data:
    # it means the spell is open.
    t["end"] = _date("Until")
    out = t[["manager", "club", "start", "end", "caretaker", "incumbent"]]
    return out[out["manager"].ne("") & out["start"].notna()].reset_index(drop=True)


def main() -> int:
    d = fetch()
    print(f"{len(d)} spells, {d.manager.nunique()} managers, "
          f"{d.club.nunique()} clubs", flush=True)
    print(f"  dates {d.start.min().date()} -> {d.start.max().date()}")
    print(f"  caretakers {int(d.caretaker.sum())}, "
          f"open spells {int(d.end.isna().sum())}")
    bad = d[d.end.notna() & d.end.lt(d.start)]
    if len(bad):
        print(f"  !! {len(bad)} spells end before they start -- not written")
        print(bad.head().to_string())
        return 3
    if OUT.exists():
        prev = pd.read_csv(OUT)
        if len(d) < len(prev):
            print(f"REFUSING to overwrite: new {len(d)} < existing {len(prev)}")
            return 3
        print(f"  was {len(prev)} rows -> now {len(d)}")
    d.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
