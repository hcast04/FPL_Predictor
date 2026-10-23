#!/bin/bash
# Progress bars for the long-running bake-off jobs. Fits 80 columns.
#
# Reads the SHAPE of each job's log rather than instrumenting the jobs, so it
# works on anything already running. Usage:
#     bash fpl2/status.sh
#     watch -n20 "BAKE_DIR=... bash fpl2/status.sh"
S="${BAKE_DIR:-$1}"
[ -z "$S" ] && S=$(ls -td /tmp/claude-*/*/*/scratchpad 2>/dev/null | head -1)
[ -d "$S" ] || { echo "no scratchpad; pass as \$1 or set BAKE_DIR"; exit 1; }
W=22

bar() {  # bar <done> <total> <label> <detail>
  local d=$1 t=$2 lbl=$3 det=$4 pct f e
  [ "${t:-0}" -le 0 ] && t=1
  [ "${d:-0}" -gt "$t" ] && d=$t
  pct=$(( d * 100 / t )); f=$(( d * W / t )); e=$(( W - f ))
  printf "  %-20s [" "$lbl"
  # printf with a format and NO args still emits the format once, so an empty
  # seq does not give an empty run of blocks -- guard both.
  [ "$f" -gt 0 ] && printf '#%.0s' $(seq 1 "$f")
  [ "$e" -gt 0 ] && printf '.%.0s' $(seq 1 "$e")
  printf "]%4d%%  %s\n" "$pct" "$det"
}
# Two traps in one function.
#  1. `pgrep -c` prints 0 AND exits non-zero on no match, so `|| echo 0` yields
#     the two-line string "0\n0" and every arithmetic test on it then fails.
#  2. `pgrep -f` matches THIS script's own shell, whose command line contains
#     the pattern -- so every job reads as "1 running" forever. Bracketing the
#     first character fixes it: the regex `[r]un_x` matches a real process's
#     `run_x`, but our own command line holds the literal `[r]un_x`, which it
#     does not match. Same trick as `ps | grep [f]oo`.
alive() {
  local pat="$1" n
  pat="[${pat:0:1}]${pat:1}"
  n=$(pgrep -fc "$pat" 2>/dev/null); echo "${n:-0}"
}
cnt()   { local n; n=$(grep -c "$1" "$2" 2>/dev/null); echo "${n:-0}"; }

echo "──────────────────────────────────────────────────────────────────────────────"
printf "  FPL bake-off  %s  load%s\n" "$(date +%H:%M:%S)" \
       "$(uptime | sed 's/.*average://' | cut -c1-18)"
echo "──────────────────────────────────────────────────────────────────────────────"

for fam in "" pos s47 gbdtus ceil; do
  tot=8; [ -z "$fam" ] && tot=16
  f="$S/logs/_progress${fam:+_$fam}.log"; [ -f "$f" ] || continue
  d=$(wc -l < "$f" | tr -d ' '); bad=$(cnt "exit=[^0]" "$f")
  r=$(alive "run_bakeoff.py ${fam:-}")
  det="$d/$tot"; [ "$bad" -gt 0 ] && det="$det !$bad"
  # only show a worker count while the job is actually incomplete
  [ "$r" -gt 0 ] && [ "$d" -lt "$tot" ] && det="$det ($r)"
  bar "$d" "$tot" "fit ${fam:-phaseA}" "$det"
done

# scorer runs: one bar each, weighted by measured wall-clock share of sections
for pair in "score_full.log|vs old ref" "score_us_gate.log|GATE vs m_full_us"; do
  log="$S/${pair%%|*}"; nm="${pair##*|}"; [ -f "$log" ] || continue
  m1=$(cnt "^   scored " "$log")
  dec=$(sed -n '/^DECOMPOSITION/,/^PER POSITION/p' "$log" | grep -cE "^[bdems]_")
  pp=$(sed -n '/^PER POSITION/,$p' "$log" | grep -cE "^[bdems]_")
  dg=$(cnt "^DIAGNOSTICS" "$log"); sk=$(cnt "^\[skipping" "$log")
  if [ "$sk" -gt 0 ]; then                     # metric1-only fast pass
    n=$(( m1 * 100 / 8 )); st="metric1 $m1/8"
  else
    n=$(( dg*15 + m1*10/8 + dec*33/17 + pp*42/11 ))
    st="d$dg m$m1/8 D$dec/17 P$pp/11"
  fi
  [ "$(cnt '^done in' "$log")" -gt 0 ] && { n=100; st="$st DONE"; }
  bar "$n" 100 "$nm" "$st"
done

for l in "$S"/opt_captain*.log; do
  [ -f "$l" ] || continue
  nm=$(basename "$l" .log | sed 's/opt_captain_*/capt /;s/^capt $/capt fpl/')
  d=$(cnt " done (" "$l"); [ "$(cnt '^done in' "$l")" -gt 0 ] && d=8
  det="$d/8"; [ "$d" -ge 8 ] && det="$det DONE"
  bar "$d" 8 "$nm" "$det"
done

l="$S/opt_horizon.log"
if [ -f "$l" ]; then
  d=$(grep -cE '^   \[' "$l"); [ "$(cnt '^done in' "$l")" -gt 0 ] && d=8
  det="$d/8"; [ "$d" -ge 8 ] && det="$det DONE"
  bar "$d" 8 "opt horizon sweep" "$det"
fi

f="$S/logs/_progress_opt2.log"
if [ -f "$f" ]; then
  d=$(wc -l < "$f" | tr -d ' '); bad=$(cnt "exit=[^0]" "$f")
  r=$(alive "bin/python fpl2/run_opt_bakeoff2.py")
  # exit!=0 here is the known G=1 reporting error; the pickles are still good
  det="$d/8"; [ "$bad" -gt 0 ] && det="$det (G=1 rpt)"
  [ "$r" -gt 0 ] && det="$det ($r)"
  bar "$d" 8 "opt2 transf+build" "$det"
fi
echo "──────────────────────────────────────────────────────────────────────────────"
