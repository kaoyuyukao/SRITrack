#!/usr/bin/env bash

SCRIPT="track.py"

SETTINGS=(setting.yaml)

for cfg in "${SETTINGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[SKIP]: $cfg"
    continue
  fi

  echo "========================================"
  echo "=== Run with config: $cfg"
  echo "========================================"
  python "$SCRIPT" -c "$cfg"
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "[ERROR] $cfg (exit code=$status)"
  else
    echo "[DONE]  $cfg"
  fi
  echo
done
