# Parallel Video Encoding During Recording

**Date:** 2026-06-25
**Status:** Approved

## Problem

After pressing the right arrow key to end an episode, `dataset.save_episode()` blocks the recording loop while videos are encoded (up to 10–30 s with `libsvtav1`). The reset phase and the next recording cannot start until encoding completes.

## Goal

Start video encoding in a background thread the moment right arrow is pressed, so the reset phase and the next recording overlap with encoding. The robot operator experiences no dead time between episodes.

## Approach: `non_blocking=True` on `save_episode` + persistent single-thread executor

### Revised recording loop (`lerobot_record.py`)

Move `save_episode()` to **before** the reset phase and pass `non_blocking=True`:

```
[Episode N records]  → right arrow pressed
save_episode(non_blocking=True)   ← Phase 1 runs on main thread (~1-2 s)
                                    Phase 2 fires in background executor
recorded_episodes += 1

[Reset phase runs]                ← overlaps with background encoding
[Episode N+1 records]             ← also overlaps if reset is short
```

**Re-record request during reset phase:** Encoding is already committed. Log
`"Re-record not possible: episode N is already being encoded. Proceeding to next episode."`
and clear `events["rerecord_episode"]`. No episode gap because `recorded_episodes` is already incremented before the reset phase.

### Two-phase split inside `save_episode`

**Phase 1 — synchronous, always runs on the calling thread (~1-2 s):**

1. `validate_episode_buffer()`
2. `_wait_image_writer()` — flush remaining PNGs from the image writer queue
3. Compute `ep_stats` via `compute_episode_stats()`
4. `_save_episode_data()` — write Parquet, update `self.latest_episode`
5. Optimistically advance `meta.info["total_episodes"] += 1` and `meta.info["total_frames"] += episode_length`, write `info.json` — this unblocks `create_episode_buffer()` for episode N+1 immediately
6. `clear_episode_buffer()` — main thread is free to record next episode

**Phase 2 — encoding work, runs in background when `non_blocking=True`:**

1. Encode camera frames into temporary video files (existing parallel `ProcessPoolExecutor` logic unchanged)
2. `_save_episode_video()` per video key — moves/concatenates video into dataset chunk files, uses `meta.latest_episode` set by previous episode's Phase 2
3. `meta._save_episode_metadata(episode_dict)` — write episode metadata to parquet (counter update already done in Phase 1; a new `skip_counter_update: bool = False` parameter gates the increment)
4. Recompute and write `meta.stats` via `aggregate_stats()` + `write_stats()`
5. Update `meta.latest_episode` — required by the next episode's `_save_episode_video()`

### Ordering guarantee

A `ThreadPoolExecutor(max_workers=1)` is stored on `LeRobotDataset`. With a single worker, Phase 2 of episode N always completes before Phase 2 of episode N+1 starts. This is essential because `_save_episode_video()` reads `meta.latest_episode` written by the previous Phase 2.

`non_blocking=True` and `batch_encoding_size > 1` are mutually exclusive. A `ValueError` is raised if both are requested.

### New fields on `LeRobotDataset`

| Field | Type | Purpose |
|---|---|---|
| `_encoding_executor` | `ThreadPoolExecutor \| None` | Created lazily on first `non_blocking=True` call |
| `_pending_encoding_future` | `Future \| None` | Tracks last submitted job so `finalize()` can wait |

### `finalize()` changes

```python
def finalize(self):
    if self._pending_encoding_future is not None:
        self._pending_encoding_future.result()   # wait; re-raises any encoding exception
    if self._encoding_executor is not None:
        self._encoding_executor.shutdown(wait=True)  # drains any queued jobs not yet started
    self._close_writer()
    self.meta._close_writer()
```

`VideoEncodingManager.__exit__()` already calls `dataset.finalize()`, so it inherits the drain automatically.

### Error handling

If Phase 2 raises (e.g., ffmpeg failure), the exception surfaces in `finalize()` before `push_to_hub`. At that point:
- Parquet data is written and valid
- `info.json` reflects the correct episode count
- Only the video files and episode metadata parquet row are missing

The dataset is recoverable by re-running encoding offline on the saved PNG frames.

## Files Changed

| File | Change |
|---|---|
| `src/lerobot/datasets/lerobot_dataset.py` | Split `save_episode()` into two phases; add `_encoding_executor`, `_pending_encoding_future`; update `finalize()` |
| `src/lerobot/datasets/lerobot_dataset.py` — `MetaData.save_episode` | Add `skip_counter_update: bool = False` parameter |
| `src/lerobot/scripts/lerobot_record.py` | Move `save_episode(non_blocking=True)` before reset phase; update re-record rejection logic |

## What Does Not Change

- Default `non_blocking=False` behavior is identical to today
- `batch_encoding_size > 1` path is untouched
- `parallel_encoding` (multi-camera ProcessPoolExecutor) is untouched
- Public API of `save_episode()` is backward compatible (new kwarg only)
