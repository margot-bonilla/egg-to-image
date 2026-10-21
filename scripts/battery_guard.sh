#!/bin/bash
# Keep the Mac alive when the charger can't supply what training draws (e.g. 60 W): pause the
# training below LOW% battery, resume it at HIGH%. Only resumes runs it paused itself.
#
#     nohup scripts/battery_guard.sh [LOW] [HIGH] > /dev/null 2>&1 &      (default 25 60)
#
# Pass --took-over to also resume a run that is already paused (e.g. paused by hand for a low battery).
cd "$(dirname "$0")/.." || exit 1
low=25 high=60 flag=logs/.battery_guard_paused
[ "${1:-}" = "--took-over" ] && { touch "$flag"; shift; }
low=${1:-$low} high=${2:-$high}
status() { echo "$(date '+%H:%M') battery guard: $*" >> logs/round2.status; }
status "started (pause <= ${low}%, resume >= ${high}%)"
while true; do
  pct=$(pmset -g batt | grep -oE '[0-9]+%' | tr -d %)
  if [ -n "$pct" ]; then
    if [ "$pct" -le "$low" ] && [ ! -f "$flag" ] && scripts/training.sh status | grep -q running; then
      scripts/training.sh pause > /dev/null && touch "$flag" && status "battery ${pct}%: paused training"
    elif [ "$pct" -ge "$high" ] && [ -f "$flag" ]; then
      scripts/training.sh resume > /dev/null; rm -f "$flag"; status "battery ${pct}%: resumed training"
    fi
  fi
  sleep 60
done
