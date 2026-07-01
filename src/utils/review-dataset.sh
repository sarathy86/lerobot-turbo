#!/bin/bash
# Usage: ./viz_episode.sh <start_episode> [total_episodes]
#   [n] next  [b] prev  [r] replay  [d] mark+next  [u] unmark  [q] quit & delete
#   'n' past the last episode -> goes straight to the delete step.
if [ -z "$1" ]; then
  echo "Usage: $0 <start_episode> [total_episodes]"
  exit 1
fi

if [ -z "$REPO_PATH" ]; then
  echo "Error: REPO_PATH is not set. Export it before running this script, e.g.:" >&2
  echo "  export REPO_PATH=/local/dataset/repo/path" >&2
  exit 1
fi

if [ ! -d "$REPO_PATH" ]; then
  echo "Error: REPO_PATH '$REPO_PATH' is not a valid directory." >&2
  exit 1
fi

REPO_ID="$(basename "$(dirname "$REPO_PATH")")/$(basename "$REPO_PATH")"
EPISODE=$1
VIZ_PID=""
declare -a TO_DELETE=()

# ---- figure out the last valid episode index ---------------------------------
TOTAL_EPISODES="$2"   # optional manual override
if [ -z "$TOTAL_EPISODES" ]; then
  INFO_JSON="$REPO_PATH/meta/info.json"
  if [ -f "$INFO_JSON" ]; then
    if command -v jq >/dev/null 2>&1; then
      TOTAL_EPISODES=$(jq -r '.total_episodes // empty' "$INFO_JSON")
    else
      TOTAL_EPISODES=$(grep -o '"total_episodes"[^,}]*' "$INFO_JSON" | grep -o '[0-9]\+' | head -1)
    fi
  fi
fi
LAST=""
[ -n "$TOTAL_EPISODES" ] && LAST=$((TOTAL_EPISODES - 1))
if [ -n "$LAST" ]; then
  echo "Dataset reports $TOTAL_EPISODES episodes (valid indices 0..$LAST)."
else
  echo "Could not read total_episodes; will detect end of dataset at runtime."
fi
# -----------------------------------------------------------------------------

kill_viewer() {
  [ -n "$VIZ_PID" ] && kill "$VIZ_PID" 2>/dev/null
  wait "$VIZ_PID" 2>/dev/null
  VIZ_PID=""
}

is_marked() {
  local e
  for e in "${TO_DELETE[@]}"; do [ "$e" = "$1" ] && return 0; done
  return 1
}
mark_episode()   { is_marked "$1" || TO_DELETE+=("$1"); }
unmark_episode() {
  local e new=()
  for e in "${TO_DELETE[@]}"; do [ "$e" = "$1" ] || new+=("$e"); done
  TO_DELETE=("${new[@]}")
}
delete_list_str() {
  printf '%s\n' "${TO_DELETE[@]}" | sort -n | uniq | paste -sd, - | sed 's/,/, /g'
}

flush_deletes() {
  [ ${#TO_DELETE[@]} -eq 0 ] && { echo "Nothing queued for deletion."; return 0; }
  local list; list=$(delete_list_str)
  echo
  echo "Episodes queued for deletion: [$list]"
  read -rp "Delete all ${#TO_DELETE[@]} episode(s)? This cannot be undone. [y/N] " confirm
  if [[ $confirm == [yY] ]]; then
    echo "Deleting ..."
    lerobot-edit-dataset \
      --repo_id "$REPO_ID" \
      --operation.type delete_episodes \
      --operation.episode_indices "[$list]"
    echo "Done."
  else
    echo "Aborted. No episodes were deleted."
  fi
}

cleanup() {   # Ctrl-C / TERM: bail without deleting
  kill_viewer
  [ ${#TO_DELETE[@]} -gt 0 ] && echo $'\nInterrupted. Queued deletes discarded: ['"$(delete_list_str)]"
  exit 0
}
trap cleanup INT TERM

end_of_dataset() {   # reached the end -> go to delete path, then exit
  kill_viewer
  echo "Reached end of dataset. Proceeding to delete step."
  flush_deletes
  exit 0
}

while true; do
  # Known-count check: don't even launch an out-of-range episode.
  if [ -n "$LAST" ] && [ "$EPISODE" -gt "$LAST" ]; then
    end_of_dataset
  fi

  if is_marked "$EPISODE"; then mark_note="  [MARKED FOR DELETION]"; else mark_note=""; fi
  echo "Visualizing episode $EPISODE${mark_note}   (queued: ${#TO_DELETE[@]})"
  lerobot-dataset-viz \
    --repo-id "$REPO_ID" \
    --episode-index "$EPISODE" \
    --display-compressed-images true &
  VIZ_PID=$!

  # Unknown-count fallback: if the viewer dies right away, treat as past-the-end.
  if [ -z "$LAST" ]; then
    sleep 2
    if ! kill -0 "$VIZ_PID" 2>/dev/null; then
      echo "Viewer for episode $EPISODE exited immediately (likely past the last episode)."
      end_of_dataset
    fi
  fi

  echo "[n] next  |  [b] prev  |  [r] replay  |  [d] mark+next  |  [u] unmark  |  [q] quit & delete"

  action=""
  while [ -z "$action" ]; do
    IFS= read -rsn1 key
    case "$key" in
      n) action="next" ;;
      b) action="prev" ;;
      r) action="replay" ;;
      u) unmark_episode "$EPISODE"; echo "Episode $EPISODE unmarked (queued: ${#TO_DELETE[@]})." ;;
      q) kill_viewer; flush_deletes; exit 0 ;;
      d)
        kill_viewer    # stop currently running episode first
        read -rp $'\nMark episode '"$EPISODE"" for deletion? [y/N] " confirm
        if [[ $confirm == [yY] ]]; then
          mark_episode "$EPISODE"
          echo "Episode $EPISODE queued (${#TO_DELETE[@]} total). Moving to next."
          action="next"
        else
          echo "Not marked."
          action="replay"
        fi
        ;;
    esac
  done

  kill_viewer

  case "$action" in
    next)
      if [ -n "$LAST" ] && [ "$EPISODE" -ge "$LAST" ]; then
        end_of_dataset      # 'n' on the last episode -> delete path
      fi
      EPISODE=$((EPISODE + 1))
      ;;
    prev)
      if [ "$EPISODE" -gt 0 ]; then EPISODE=$((EPISODE - 1)); else echo "Already at episode 0."; fi
      ;;
    replay) : ;;
  esac
done
