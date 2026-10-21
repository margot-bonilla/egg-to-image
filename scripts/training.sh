#!/bin/bash
# Pause, resume or show the local training runs (scripts/train_solver.py and its data workers).
#
#     scripts/training.sh pause | resume | status
#
# Pausing freezes the processes in place (SIGSTOP): the GPU is free at once (the screen and
# other apps get it back) and nothing is lost; resume continues exactly where it stopped.
# The queue scripts in logs/ just wait for the paused run.
set -u
mains=$(pgrep -f "scripts/train_solver.py")
case "${1:-status}" in
  pause | resume)
    [ -z "$mains" ] && { echo "No training run."; exit 0; }
    sig=$([ "$1" = pause ] && echo STOP || echo CONT)
    for main in $mains; do
      # Workers first on resume, so the main process finds them running.
      [ "$sig" = CONT ] && kill -CONT $(pgrep -P "$main") 2>/dev/null
      kill -"$sig" "$main"
      [ "$sig" = STOP ] && kill -STOP $(pgrep -P "$main") 2>/dev/null
    done
    echo "$([ "$1" = pause ] && echo Paused || echo Resumed) $(echo $mains | wc -w | tr -d ' ') training run(s)."
    ;;
  status)
    [ -z "$mains" ] && { echo "No training run."; exit 0; }
    for main in $mains; do
      state=$(ps -o stat= -p "$main")
      out=$(ps -o command= -p "$main" | awk '{print $4}')
      echo "$out: $([[ $state == T* ]] && echo PAUSED || echo running), started $(ps -o lstart= -p "$main")"
      log=$(ls -t logs/"$(basename "$out")"_train.log 2>/dev/null | head -1)
      [ -n "$log" ] && grep "^epoch" "$log" | tail -1
    done
    ;;
  *) echo "usage: $0 pause | resume | status"; exit 1 ;;
esac
