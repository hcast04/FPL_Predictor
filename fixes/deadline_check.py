#!/usr/bin/env python3
"""
Pre-deadline re-run and analysis.

Does the mechanical work for a scheduled pre-deadline check so the reporting
agent only has to read and interpret the output:

  1. snapshot the current recommendation as a baseline
  2. re-execute the notebook headlessly (picks up fresh FPL prices and news)
  3. report any execution errors
  4. print the execution card
  5. diff the new recommendation against the baseline
  6. check live availability for every player in the recommended XI, bench and
     incoming transfers -- the single highest-value check before a deadline

Run:  .venv/bin/python fixes/deadline_check.py            # full run
      .venv/bin/python fixes/deadline_check.py --no-run   # analyse only
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
JUPYTER = ROOT / ".venv" / "bin" / "jupyter"
NB = ROOT / "FPL_Predictor_v2.6.ipynb"
EXECUTED = ROOT / "FPL_Predictor_v2.6_executed.ipynb"
REC = ROOT / "outputs" / "latest_weekly_recommendation.json"
BASELINE = Path("/tmp/fpl_deadline_baseline.json")

BAR = "=" * 72


def say(*a):
    print(*a, flush=True)


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def snapshot_baseline():
    rec = load(REC)
    if rec:
        BASELINE.write_text(json.dumps(rec), encoding="utf-8")
        say(f"baseline captured from {REC.name}")
    else:
        say("! no existing recommendation to use as a baseline")
    return rec


def run_notebook() -> bool:
    say(f"\n{BAR}\nRE-EXECUTING THE NOTEBOOK (this takes 10-45 minutes)\n{BAR}")
    before = EXECUTED.stat().st_mtime if EXECUTED.exists() else 0
    proc = subprocess.run(
        [str(JUPYTER), "nbconvert", "--to", "notebook", "--execute",
         "--allow-errors", "--ExecutePreprocessor.timeout=5400",
         "--output", EXECUTED.name, str(NB)],
        cwd=ROOT, capture_output=True, text=True,
    )
    ok = EXECUTED.exists() and EXECUTED.stat().st_mtime > before
    say(f"nbconvert exit={proc.returncode}  output refreshed={ok}")
    if not ok:
        say("! the notebook did not produce a new output file")
        say(proc.stderr[-1500:])
    return ok


def report_errors() -> int:
    nb = load(EXECUTED)
    if not nb:
        say("! could not read the executed notebook")
        return -1
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = [o for c in code for o in c.get("outputs", [])
            if o.get("output_type") == "error"]
    say(f"\ncells={len(code)}  errors={len(errs)}")
    for o in errs[:5]:
        say(f"  !! {o.get('ename')}: {str(o.get('evalue'))[:300]}")
    return len(errs)


def print_card():
    nb = load(EXECUTED)
    if not nb:
        return
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        text = ""
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                text += "".join(o.get("text", []))
            elif "data" in o and "text/plain" in o["data"]:
                text += "".join(o["data"]["text/plain"])
        if "EXECUTE THIS" in text:
            say(f"\n{BAR}\nEXECUTION CARD\n{BAR}")
            say(text)
            return
    say("\n! no execution card found in the notebook outputs")


def diff_vs_baseline():
    before = load(BASELINE)
    after = load(REC)
    say(f"\n{BAR}\nCHANGE VS THE EARLIER RECOMMENDATION\n{BAR}")
    if not before or not after:
        say("! cannot compare - missing baseline or new recommendation")
        return
    b, a = before.get("summary", {}), after.get("summary", {})
    fields = ["gameweek", "primary_action", "chip_action", "captain",
              "vice_captain", "expected_current_score", "decision_confidence"]
    changed = []
    for k in fields:
        same = b.get(k) == a.get(k)
        if not same:
            changed.append(k)
        say(f"  {k:24s} {'same' if same else 'CHANGED'}")
        if not same:
            say(f"      was: {b.get(k)}")
            say(f"      now: {a.get(k)}")
    say(f"\n  => {'NOTHING CHANGED' if not changed else 'CHANGED: ' + ', '.join(changed)}")


def check_availability():
    rec = load(REC)
    if not rec:
        return
    names = set()
    for key in ("starting_xi", "bench"):
        for row in rec.get(key) or []:
            n = row.get("web_name")
            if n:
                names.add(str(n))
    action = str((rec.get("summary") or {}).get("primary_action") or "")
    for part in action.replace("→", "->").split(";"):
        if "->" in part:
            names.add(part.split("->")[-1].strip())

    say(f"\n{BAR}\nLIVE AVAILABILITY FOR EVERY RECOMMENDED PLAYER\n{BAR}")
    try:
        req = urllib.request.Request(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            headers={"User-Agent": "Mozilla/5.0"})
        data = json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as exc:
        say(f"! could not reach the FPL API: {exc}")
        return
    teams = {t["id"]: t["short_name"] for t in data["teams"]}
    flagged, clean = [], []
    for e in data["elements"]:
        if e["web_name"] in names:
            row = (e["web_name"], e["status"], e["chance_of_playing_next_round"],
                   teams[e["team"]], e["news"][:60])
            (clean if e["status"] == "a" else flagged).append(row)
    if flagged:
        say("  FLAGGED:")
        for n, s, c, t, news in sorted(flagged):
            say(f"    {n:18s} {t:4s} status={s} chance={c}  {news}")
    else:
        say("  no flagged players among those named")
    say(f"\n  clean: {', '.join(sorted(n for n, *_ in clean))}")
    say("\n  NOTE: match a flagged name against the XI above - a flagged BENCH")
    say("  player matters much less than a flagged STARTER.")


def main():
    say(f"{BAR}\nFPL PRE-DEADLINE CHECK\n{BAR}")
    snapshot_baseline()
    if "--no-run" not in sys.argv:
        if not run_notebook():
            say("\n! ABORTING ANALYSIS - the notebook did not complete")
            say("! fall back to the previous recommendation and tell the user")
            return 1
    n_err = report_errors()
    print_card()
    diff_vs_baseline()
    check_availability()
    say(f"\n{BAR}\nDONE  (execution errors: {n_err})\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
